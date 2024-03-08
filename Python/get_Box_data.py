import glob, os

os.chdir("/Box/Gait Study/")
for file in glob.glob("*.xlsx"):
    print(file)