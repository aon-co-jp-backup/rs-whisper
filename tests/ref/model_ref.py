# Independent Python reader of a ggml file; prints the same summary as rs-whisper-model.
import struct, sys
b = open(sys.argv[1], 'rb').read(); p = 0
def rd(fmt):
    global p
    v = struct.unpack_from(fmt, b, p); p += struct.calcsize(fmt); return v
assert rd('<I')[0] == 0x67676d6c
h = rd('<11i'); print('hparams', *h)
nm, nf = rd('<2i'); fl = struct.unpack_from('<%df' % (nm*nf), b, p); p += nm*nf*4
print('filters', nm, nf, 'sum=%.6f' % sum(fl))
nv, = rd('<i'); vb = 0
for _ in range(nv):
    l, = rd('<I'); vb += l; p += l
print('vocab', nv, 'bytes=%d' % vb)
L = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34)}
ts = []
while p < len(b):
    nd, ln, tt = rd('<3i'); dims = list(rd('<%di' % nd)) if nd else []
    name = b[p:p+ln].decode(); p += ln
    ne = 1
    for d in dims: ne *= d
    blk, bs = L[tt]; nb = ne // blk * bs; p += nb
    ts.append('T %s %s %d %d' % (name, dims, tt, nb))
print('tensors', len(ts))
for t in ts: print(t)
