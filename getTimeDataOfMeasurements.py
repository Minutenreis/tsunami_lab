import sys

if (len(sys.argv) >= 2):
    path = sys.argv[1]
else:
    path = "tsunami_lab_reis.output"
with open(path, "r") as file:
    data = file.readlines()

# Get the time data of the measurements
timeLines = [line for line in data if line.startswith("calc time per cell and iteration:")]
timeData = [float(line.strip("calc time per cell and iteration:ns\n")) for line in timeLines]
print(timeData)