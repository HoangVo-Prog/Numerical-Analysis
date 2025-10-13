# Run this file: bash run_euler.sh

theta=1

F='x*y - x**3'
EXACT='x**2 + 2 - np.exp(x**2/2)'  

python euler_cli.py --py "$F" -a 0 -b 2 -N 50 \
  --method euler_forward --theta $theta \
  --plot --save-plot euler_forward.png \
  --exact "$EXACT"

F='x*y - x**3'
EXACT='x**2 + 2 - 5*np.exp(x**2/2 - 2)'
python euler_cli.py --py "$F" -a 0 -b 2 -N 50 \
  --method euler_backward --theta $theta \
  --plot --save-plot euler_backward.png --exact "$EXACT"

