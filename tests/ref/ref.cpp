// whisper.cpp(MIT, (c) The ggml authors)の dft/fft/log_mel を元のまま取り出した基準実装(単一スレッド)
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cassert>
#define WHISPER_N_FFT 400
#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_HOP_LENGTH 160
struct whisper_filters { int n_mel; int n_fft; std::vector<float> data; };
struct whisper_mel { int n_len; int n_len_org; int n_mel; std::vector<float> data; };
#define SIN_COS_N_COUNT WHISPER_N_FFT
namespace {
struct whisper_global_cache {
    // In FFT, we frequently use sine and cosine operations with the same values.
    // We can use precalculated values to speed up the process.
    float sin_vals[SIN_COS_N_COUNT];
    float cos_vals[SIN_COS_N_COUNT];

    // Hann window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    float hann_window[WHISPER_N_FFT];

    whisper_global_cache() {
        fill_sin_cos_table();
        fill_hann_window(sizeof(hann_window)/sizeof(hann_window[0]), true, hann_window);
    }

    void fill_sin_cos_table() {
        for (int i = 0; i < SIN_COS_N_COUNT; i++) {
            double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
            sin_vals[i] = sinf(theta);
            cos_vals[i] = cosf(theta);
        }
    }

    void fill_hann_window(int length, bool periodic, float * output) {
        int offset = -1;
        if (periodic) {
            offset = 0;
        }
        for (int i = 0; i < length; i++) {
            output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
        }
    }
} global_cache;
}

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const float* in, int N, float* out) {
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT); // t = 2*M_PI*k*n/N
            re += in[n]*global_cache.cos_vals[idx]; // cos(t)
            im -= in[n]*global_cache.sin_vals[idx]; // sin(t)
        }

        out[k*2 + 0] = re;
        out[k*2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
static void fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    const int half_N = N / 2;
    if (N - half_N*2 == 1) {
        dft(in, N, out);
        return;
    }

    float* even = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i]= in[2*i];
    }
    float* even_fft = out + 2 * N;
    fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2*i + 1];
    }
    float* odd_fft = even_fft + N;
    fft(odd, half_N, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < half_N; k++) {
        int idx = k * sin_cos_step; // t = 2*M_PI*k/N
        float re = global_cache.cos_vals[idx]; // cos(t)
        float im = -global_cache.sin_vals[idx]; // sin(t)

        float re_odd = odd_fft[2*k + 0];
        float im_odd = odd_fft[2*k + 1];

        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        out[2*(k + half_N) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + half_N) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}
static void log_mel_spectrogram_worker_thread(int ith, const float * hann, const std::vector<float> & samples,
                                              int n_samples, int frame_size, int frame_step, int n_threads,
                                              const whisper_filters & filters, whisper_mel & mel) {
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);

    int n_fft = filters.n_fft;
    int i = ith;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    assert(n_fft == 1 + (frame_size / 2));

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hann window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in.data(), frame_size, fft_out.data());

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;
            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum +=
                        fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                        fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                        fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                        fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }
            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }
            sum = log10(std::max(sum, 1e-10));
            mel.data[j * mel.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
}

static void run(const std::vector<float>& s, const whisper_filters& filters, whisper_mel& mel) {
    const int n_samples = s.size(), frame_size = WHISPER_N_FFT, frame_step = WHISPER_HOP_LENGTH;
    const float * hann = global_cache.hann_window;
    int64_t stage_1_pad = WHISPER_SAMPLE_RATE * 30, stage_2_pad = frame_size / 2;
    std::vector<float> samples_padded(n_samples + stage_1_pad + stage_2_pad * 2);
    std::copy(s.begin(), s.end(), samples_padded.begin() + stage_2_pad);
    std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);
    const int64_t n_reflect = std::min<int64_t>(stage_2_pad, std::max<int64_t>(0, (int64_t) n_samples - 1));
    std::reverse_copy(s.data() + 1, s.data() + 1 + n_reflect, samples_padded.begin() + (stage_2_pad - n_reflect));
    mel.n_len = (samples_padded.size() - frame_size) / frame_step;
    mel.data.resize(mel.n_mel * mel.n_len);
    log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples + stage_2_pad, frame_size, frame_step, 1, filters, mel);
    double mmax = -1e20;
    for (size_t i = 0; i < mel.data.size(); i++) if (mel.data[i] > mmax) mmax = mel.data[i];
    mmax -= 8.0;
    for (size_t i = 0; i < mel.data.size(); i++) { if (mel.data[i] < mmax) mel.data[i] = mmax; mel.data[i] = (mel.data[i] + 4.0)/4.0; }
}
static std::vector<float> rd(const char*p){FILE*f=fopen(p,"rb");std::vector<float> v;float x;while(fread(&x,4,1,f)==1)v.push_back(x);fclose(f);return v;}
int main(int c,char**a){
    auto s=rd(a[1]); whisper_filters f; f.n_mel=atoi(a[3]); f.n_fft=201; f.data=rd(a[2]);
    whisper_mel m; m.n_mel=f.n_mel; run(s,f,m);
    FILE*o=fopen(a[4],"wb"); fwrite(m.data.data(),4,m.data.size(),o); fclose(o); return 0; }
