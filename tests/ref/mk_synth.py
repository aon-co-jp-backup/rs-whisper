# Builds a synthetic ggml file (real header of the test model + f32/f16/q8_0/q4_0 tensors)
# to exercise the tensor reader. Usage: mk_synth.py <base.bin> <out.bin>
import struct, sys, random
src = open(sys.argv[1], 'rb').read(); out = bytearray(src)
random.seed(1)
def tensor(name, dims, tt):
    ne = 1
    for d in dims: ne *= d
    L = {0: (1, 4), 1: (1, 2), 2: (32, 18), 8: (32, 34)}[tt]
    nb = ne // L[0] * L[1]
    nm = name.encode()
    return (struct.pack('<3i', len(dims), len(nm), tt) + struct.pack('<%di' % len(dims), *dims)
            + nm + bytes(random.randrange(256) for _ in range(nb)))
out += tensor('encoder.positional_embedding', [512, 1500], 0)
out += tensor('decoder.token_embedding.weight', [512, 64], 1)
out += tensor('encoder.blocks.0.attn.query.weight', [512, 512], 8)
out += tensor('decoder.blocks.0.attn.key.weight', [256, 4], 2)
out += tensor('scalar', [], 0)
open(sys.argv[2], 'wb').write(out)
