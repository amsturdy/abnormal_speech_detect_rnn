
# config
classes = ["backgrouds",  "gunshot", "glassbreak",  "babycry"]
threshold = [0, 0.5, 0.5, 0.5]
ig = [0, 5, 4, 12]
mg = [0, 12, 12, 12]
num_classes = 4
sample_rate = 16000.
n_window = 1024
n_overlap = 360      # ensure 240 frames in 10 seconds
max_len = 240        # sequence max length is 10 s, 240 frames. 
step_time = float(n_window - n_overlap) / sample_rate
