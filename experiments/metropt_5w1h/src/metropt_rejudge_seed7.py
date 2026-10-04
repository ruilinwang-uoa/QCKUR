"""One-off: L2/L3 rejudge pass at seed 7 (r9 zero-claim sweep, pass 2).
AI4I precedent: seed variation recovers part of the stubborn zero-claim
failures that same-seed retries leave behind."""
import metropt_l2l3_eval as m

m.SEED = 7
m.run_layer("l2", workers=4)
m.run_layer("l3", workers=4)
