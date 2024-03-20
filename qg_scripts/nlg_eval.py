from nlgeval import compute_individual_metrics

import sys

if len(sys.argv) != 3:
  print("hyp_file, ref_file are 1st and 2nd arguments") 
  sys.exit(1)

metrics_dict = compute_individual_metrics([sys.argv[2]], sys.argv[1])
print(metrics_dict)
