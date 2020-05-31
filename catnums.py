# computing with binary trees (with empty leaves)
 
import visuals
import numpy as np
# bijection from N to T (standing for the set of binary trees)
def t(n) :
  if n is 0 : return ()
  else :
    assert n>0
    x,y = to_pair_with(nparity(n),n)
    ys = t(y)
    if x is 0 :
      zs=ys
    else :
      return  (t(x-1),ys)

# inverse of t, from trees to N
def n(t) :
 if t is () : return 0
 else :
   x,xs=t
   b = tparity(t)
   if b==0 : return 2**(n(x)+1)*n(xs)
   else : return 2**(n(x)+1)*(n(xs)+1)-1

# parity, in N 
def nparity(n) : return n % 2

# split into count of 0s/1s ending it and the rest of a number  
def to_pair_with(b,z) :
  x=0
  while z>0 and nparity(z)==b :
    z=(z-b)//2
    x+=1
  return (x,z)  
    
# parity in T
def tparity(xs) :
  l=0
  while xs :
    x,ys=xs 
    xs=ys 
    l+=1 
  return l%2

# successor, from T to T without empty tree  
def s(t) :
  one = (),()
  if t is () : return one
  else :
    x,y=t
    if y is () : return x,one
    else :
      u,v=y
      b=tparity(t)
      if b is 0 :
        if x is () : return s(u),v
        else : 
          return (),(s_(x),y)      
      else :
        if u is () and v is not ():
          w,z=v
          return x,(s(w),z)
        else :
          return x, ((),(s_(u),v))

# predecessor, from T without empty tree  to T         
def s_(t) :
  one = (),()
  x,y=t
  if t == one : return ()
  elif y == one : return x,()
  else :
    b=tparity(t)
    if b is 0 :
      u,v=y
      if u is () and v is not () :
        w,z=v
        return x,(s(w),z)
      else :
        return x, ((),(s_(u),v))     
    else :
      if x is () :
        u,v=y
        return s(u),v
      else :
        return (),(s_(x),y)

# double, from T to T     
def db(t) :
  if t is () : return ()
  else : 
    if tparity(t) is 0 :
      x,y=t
      return s(x),y
    else :
      return (),t

# half rounded down, from T to T  
def hf(t) :
  if t is () : return ()
  else :
    x,y=t
    if x is () : return y
    else : return s_(x),y  

# power of 2, from T to powers of 2 in T
def exp2(t) :
  one = (),()
  if t is () : return one
  else : return s_(t),one

# integer log of 2, from powers of 2 in T, to T
def log2(t) :
  one = (),()
  if t == one : return ()
  else :
    x,y=t
    assert y==one
    return s(x)

def tsize(t) :
  if not t : return 1
  else :
    return 1+sum(tsize(x) for x in t)

# ML rose-tree interface

chars="[] "

L,R,S=chars



def to_rose_tree(t) :
  if t==() : return []
  l,r=t
  m = []
  while True:
    x = to_rose_tree(l)
    m.append(x)
    if r==() : return m
    l,r=r

def rose2str(t) :
  def f(t) :
    if t==[] : return ""
    xs=[]
    for x in t :
      xs.append(L)
      xs.append(f(x))
      xs.append(R)
    return "".join(xs)
  return L+f(t)+R

def to_rose_str(t) :
  return rose2str(to_rose_tree(t))

def go() :
  for i in range(16) :
    x=t(i)
    y=to_rose_tree(x)
    print('B',x)
    print('M',y)
    s=rose2str(y)
    print('S',s)
    print('')


# ML char string binary tree interface
def to_encoded_str(t) :
  def f(t) :
    if t==() : return [L]
    x,y=t
    return f(x)+f(y)+[R]
  return "".join(f(t))


# generic pair

to_str=to_rose_str

def pad_to_str(max,xs) :
  l=len(xs)
  m=max-l
  return xs + (S*m)

def nats_from_str(t,k) :
  pss=[]
  m=0
  x = t
  for _ in range(k) :
    sx=s(x)
    bs=to_str(x)
    cs=to_str(sx)
    pss.append((bs,cs))
    m=max(m,len(bs),len(cs))
    x=sx
  m+=2
  zipped = [(pad_to_str(m,bs),pad_to_str(m,cs)) for (bs,cs) in pss]
  unzipped = zip(*zipped)
  return tuple(unzipped)

def to_dataset(k,fname) :
  qs, rs = nats_from_str(t(0), k)
  with open(fname,'w') as f :
    for i in range(len(qs)) :
      q=qs[i]
      r=rs[i]
      s=q.strip()+":"+r.strip()
      print(s,file=f)

def mats_from(n,k) :
  pss=[]
  tree=t(n)
  l=tsize(tree)
  for _ in range(k) :
    stree=s(tree)
    pss.append([tree,stree])
    l=max(l,tsize(stree))
    tree=stree


  mss=[(tree2mat(x,l),tree2mat(y,l,l+1)) for x,y in pss]
  unzipped = zip(*mss)
  return unzipped


# visuals

# draw n as a tree
def st(n) :
  tnum=t(n)
  dt=decorate(tnum)
  visuals.showTree(dt)

# draw n as a digraph
def sd(n) :
  tnum=t(n)
  dt=decorate(tnum)
  visuals.showDag(dt) 
  
def decorate(t) :
    if t is () : return ('0',)
    else :
      x,y=t
      return (str(n(t)),decorate(x),decorate(y))

# tree to matrix conversion

def plus(xs,ys) :
  return tuple(map((lambda x,y : x+y),xs,ys))

def h(t,size=None) :
  i=0
  if not size : size=tsize(t)
  z=(0,)*size

  def e(i):
    xs = [0] * size
    xs[i] = 1
    return xs

  def walk(t,p) :
    nonlocal i
    v = plus(e(i),p)
    i += 1
    yield v
    for x in t :
      yield from walk(x,v)

  yield from walk(t,z)

def tree2mat(t,size=None,pad=None) :
  xs=list(h(t,size=size))
  z=(0,)*len(xs[0])
  if not pad : pad=size
  for _ in range(len(xs),pad) :
    xs.append(z)
  return np.array(xs)


def m(k,size=None) :
  return tuple(h(t(k),size=size))

# tests

def ppp(*xs) :
  print('DEBUG:',*xs)

def t0(k) :
  tree = t(k)
  print(tree)
  for x in tree2mat(tree):
    print(x)
  print('')

def t1():
  for x in range(10000):
    y = n(s_(s(t(x))))
    if x != y: print(x, y)
  for x in range(1, 10000):
    y = n(s(s_(t(x))))
    if x != y: print(x, y)
  print('done')


def t2():
  for x in range(10000):
    y = n(hf(db(t(x))))
    if x != y: print(x, y)
  for x in range(0, 10000, 2):
    y = n(db(hf(t(x))))
    if x != y: print(x, y)
  print('done')


def s1():
    bs,cs = nats_from_str(t(0),5)
    print('Q:',bs)
    print('A:',cs)
    print('')

def dgo() : to_dataset(2**16,'cats.txt')

def mgo():
  # t0(42)
  # t0(256)
  r, rs = mats_from(0, 5)
  print(len(list(r)))
  for c in rs:
    print(c)
    print('')

if __name__=="__main__" :
  pass

