#!/bin/sh
# §17 長走 parity 実測(shell のみ・新規 Rust/C/Swift/Python 零)。
# fieldrun(Q20/Q30) 対 FLDJ_REF=1 replay_fldj(wave_step_reference f32)。
set -eu
cd "$(dirname "$0")"
W=${W:-32}; H=${H:-32}; export W H
REPLAY=${REPLAY:-../../../target/debug/examples/replay_fldj}
FIELDC=${FIELDC:-../fieldc}
work=./lp-work.$$
mkdir "$work"; trap 'rm -rf "$work"' EXIT HUP INT TERM
printf '%-6s %-6s %-13s %-13s %-13s %-13s %s\n' N sat RMSREF RMSREL MEDREL P95REL note
for N in ${NLIST:-1 10 50 100 200}; do
  STEPN=$N ./gen_fld_a7.sh "$work/a.fld"
  "$FIELDC" "$work/a.fld" "$work/a.fldj"
  ./fieldrun "$work/a.fldj" "$work/a.flro" 16384 >/dev/null
  sat=$(od -An -tu4 -j24 -N4 "$work/a.flro" | tr -d ' ')
  od -An -v -td4 -j32 "$work/a.flro" | tr -s ' ' '\n' | grep -v '^$' > "$work/q.txt"
  FLDJ_REF=1 "$REPLAY" "$work/a.fldj" > "$work/r.txt"
  paste "$work/q.txt" "$work/r.txt" | awk -v N="$N" -v sat="$sat" '
    { q=$1/1048576.0; r=$2+0; d=q-r; a=(d<0?-d:d); s2+=d*d; r2+=r*r; n++; A[n]=a }
    END{ rms=sqrt(s2/n); rref=sqrt(r2/n);
      for(i=1;i<=n;i++) for(j=i+1;j<=n;j++) if(A[j]<A[i]){t=A[i];A[i]=A[j];A[j]=t}
      med=A[int((n+1)/2)]; p95=A[int(0.95*n+0.5)];
      printf "%-6s %-6s %-13.6e %-13.6e %-13.6e %-13.6e cells=%d\n",N,sat,rref,rms/rref,med/rref,p95/rref,n }'
done
