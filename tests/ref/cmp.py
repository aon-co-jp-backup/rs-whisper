import array,sys
a=array.array('f');b=array.array('f')
a.frombytes(open(sys.argv[1],'rb').read());b.frombytes(open(sys.argv[2],'rb').read())
assert len(a)==len(b),(len(a),len(b))
exact=sum(1 for x,y in zip(a,b) if x==y); mx=max(abs(x-y) for x,y in zip(a,b))
print(f"values={len(a)} exact={exact} max_abs_diff={mx:.3e}")
sys.exit(0 if mx<1e-5 else 1)
