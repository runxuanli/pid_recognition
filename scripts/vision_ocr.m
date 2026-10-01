#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <ImageIO/ImageIO.h>

static void Die(NSString *message) {
    fprintf(stderr, "%s\n", message.UTF8String);
    exit(2);
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc != 2) Die(@"usage: vision_ocr IMAGE");
        NSString *path = [NSString stringWithUTF8String:argv[1]];
        NSURL *url = [NSURL fileURLWithPath:path];
        CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
        if (!source) Die([NSString stringWithFormat:@"could not read image: %@", path]);
        CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        CFRelease(source);
        if (!image) Die([NSString stringWithFormat:@"could not decode image: %@", path]);

        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.usesLanguageCorrection = NO;
        request.recognitionLanguages = @[@"en-US"];
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:image options:@{}];
        NSError *error = nil;
        BOOL ok = [handler performRequests:@[request] error:&error];
        CGImageRelease(image);
        if (!ok) Die([NSString stringWithFormat:@"Vision OCR failed: %@", error]);

        NSMutableArray *items = [NSMutableArray array];
        for (VNRecognizedTextObservation *observation in request.results) {
            NSArray<VNRecognizedText *> *candidates = [observation topCandidates:5];
            VNRecognizedText *candidate = [candidates firstObject];
            if (!candidate) continue;
            CGRect box = observation.boundingBox;
            NSMutableArray *alternatives = [NSMutableArray array];
            for (VNRecognizedText *alternative in candidates) {
                [alternatives addObject:@{
                    @"text": alternative.string,
                    @"confidence": @(alternative.confidence)
                }];
            }
            [items addObject:@{
                @"text": candidate.string,
                @"confidence": @(candidate.confidence),
                @"alternatives": alternatives,
                @"x": @(box.origin.x),
                @"y": @(box.origin.y),
                @"width": @(box.size.width),
                @"height": @(box.size.height)
            }];
        }
        NSData *json = [NSJSONSerialization dataWithJSONObject:items options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&error];
        if (!json) Die([NSString stringWithFormat:@"could not encode OCR results: %@", error]);
        [[NSFileHandle fileHandleWithStandardOutput] writeData:json];
        [[NSFileHandle fileHandleWithStandardOutput] writeData:[@"\n" dataUsingEncoding:NSUTF8StringEncoding]];
    }
    return 0;
}
