#!/usr/bin/python3
import jetson_inference
import jetson_utils

import argparse
import sys
from jetson_inference import detectNet, imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log, cudaDrawRect, cudaAllocMapped, cudaResize
# import time
# parse the command line

parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())
parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--ssl-key", type=str, default='key.pem', help="path to SSL key file")
parser.add_argument("--ssl-cert", type=str, default='cert.pem', help="path to SSL certificate file")
parser.add_argument("--labels", type=str, default="project/model/labels.txt", help="path to the labels file")
parser.add_argument("--input_blob", type=str, default="input_0", help="name of the input blob")
parser.add_argument("--output_blob", type=str, default="output_0", help="name of the output blob")
parser.add_argument("--network_classifier", type=str, default="project/model/resnet18.onnx", help="model to use, can be:  googlenet, resnet-18, etc. (see --help for others)")
# parser.add_argument("--network_detector", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")


opt = parser.parse_args()

# Load the network
# detector   = detectNet(opt.network_detector,  sys.argv, opt.threshold)
classifier = imageNet(model=opt.network_classifier,
                      labels=opt.labels,
                      input_blob=opt.input_blob,
                      output_blob=opt.output_blob)


input = videoSource(opt.input, argv=sys.argv)
output = videoOutput(opt.output, argv=sys.argv)
font = cudaFont()

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue

    # detections = detector.Detect(img, overlay=opt.overlay)

    # image classification (whole frame)
    classID, confidence = classifier.Classify(img)
    classLabel = classifier.GetClassLabel(classID)
    confidence *= 100.0

    # overlay classifier results (top-left corner)
    font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}",
                     x=5, y=5 + 1 * (font.GetSize() + 5),
                     color=font.White, background=font.Gray40)


    # for detection in detections:
    #     print(detection)

    print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")

    # render
    output.Render(img)
    output.SetStatus(
        # f"{opt.network_detector} {detector.GetNetworkFPS():.0f} FPS | "
        f"{opt.network_classifier} {classifier.GetNetworkFPS():.0f} FPS"
    )

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
