camera calibration

use calibration.py
It will cerate numbers for camera calibration based on the jpg files.
Then it will save those numbers into json files.
Then it will undistort the jpg files and show them to you.

If you want to use your own jpg files:
1. print https://docs.opencv.org/3.4/pattern.png
2. This is an 10x7 chess board pattern, but opencv will use only "the inner"
side of the pattern, so you have to set the code 9x6. If it is 9x6 or 6x9 it depends on how
to use your camera.
3. Save many jpg files. The chessboadr should be as big as possible. The "inner" part should be
visible. Rotate the pattern. Pull it to the side, move it.
4. Run calibrate.py
