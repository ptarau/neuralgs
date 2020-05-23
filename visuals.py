from graphviz import Digraph
import random

def showTree(t) :
  g=Digraph()
  i=0
  
  def label(x) : 
    if isinstance(x,tuple) :
      return x[0]
    else :
      return str(x)
   
  def link(a,i,b,j) : 
    si=str(i)
    sj=str(j)
    g.node(si,a)
    g.node(sj,b)
    g.edge(si,sj)
        
  def st(a) :   
    nonlocal i
  
    if isinstance(a,tuple) :
      op=a[0]
      i0=i
      for x in a[1:] :
        l=label(x)
        i+=1
        link(op,i0,l,i)     
        st(x)
     
  st(t)
  g.view()   

  
def showDag(t) :
  g=Digraph()
  
  def label(x) : 
    if isinstance(x,tuple) :
      return str(x[0])
    else :
      return str(x)
   
  def link(a,b) : 
    g.edge(a,b)
        
  def st(a) :   
  
    if isinstance(a,tuple) :
      op=a[0]
      for x in a[1:] :
        l=label(x)
        link(op,l)     
        st(x)
     
  st(t)
  g.view()   
  

# small tests

  
def gt() :
  c=random.choice([0,1])
  
  t=('l', ('l', ('a', ('a', 0, ('l', 3)), ('a', ('l', 0), 0))))
  print(t)
  if c==0 :
    showDag(t)
  else :
    showTree(t)
  
  