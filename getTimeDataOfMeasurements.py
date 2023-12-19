
with open("tsunami_lab_reis.output", "r") as file:
    data = file.readlines()

# Get the time data of the measurements
timeLines = [line for line in data if line.startswith("calc time per cell and iteration:")]
timeData = [float(line.strip("calc time per cell and iteration:ns\n")) for line in timeLines]
print(timeData)