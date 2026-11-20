# 試験入力を作る: samples.f32(jfk.wav 16bit mono 16k)と、三角形のメルフィルタ(n_mel=80)
import struct, sys, array
wav = open(sys.argv[1], 'rb').read()
i = wav.find(b'data'); n = struct.unpack('<I', wav[i+4:i+8])[0]
pcm = array.array('h'); pcm.frombytes(wav[i+8:i+8+n])
s = array.array('f', [x/32768.0 for x in pcm]); open('samples.f32','wb').write(s.tobytes())
n_mel, n_fft = 80, 201
f = array.array('f')
for m in range(n_mel):
    c0, c1, c2 = m*n_fft/(n_mel+1), (m+1)*n_fft/(n_mel+1), (m+2)*n_fft/(n_mel+1)
    for k in range(n_fft):
        v = 0.0
        if c0 < k <= c1: v = (k-c0)/(c1-c0)
        elif c1 < k < c2: v = (c2-k)/(c2-c1)
        f.append(v*0.05)
open('filters.f32','wb').write(f.tobytes())
