#!/usr/bin/env python
import cv2
import sys

def convert(filename):
    vidcap = cv2.VideoCapture(filename)
    count = 0

    while True:
        success, image = vidcap.read()
        if not success:
            break
        cv2.imwrite("%s_%d.jpg" % (filename, count), image)
        count+=1

def main():
    if len(sys.argv) < 2:
        print("error please provide a mp4 file to split")
        return

    filename = sys.argv[1]
    convert(filename)

    return True

if __name__ == '__main__':
    if not main():
        sys.exit(1)