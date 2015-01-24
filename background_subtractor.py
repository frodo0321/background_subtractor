import cv2
import numpy as np
import video

class BackgroundSubtractor(object):
    def __init__(self, reference_frame, static_convergence_rate=.015, difference_threshold=.01, total_difference_threshold=.5):
        if reference_frame.dtype=='uint8':
            self.reference_frame=reference_frame.astype(np.float32)/255
        else:
            self.reference_frame=reference_frame
        self.frame=None
        self.corrector=np.zeros_like(reference_frame, dtype=np.float32)
        self.difference_mask=np.zeros_like(reference_frame, dtype=np.float32)
        self.previous_divverence_mask=np.zeros_like(reference_frame, dtype=np.float32)
        self.inverse_difference_mask=np.ones_like(reference_frame, dtype=np.float32)
        self.corrector_mask=np.zeros_like(reference_frame, dtype=np.float32)

        self.static_convergence_rate=static_convergence_rate
        self.difference_threshold=difference_threshold
        self.total_difference_threshold=total_difference_threshold



    def apply(self, frame):
        if frame.dtype=='uint8':
            self.frame=frame.astype(np.float32)/255.
        else:
            self.frame=frame
        difference=np.abs(self.frame-self.reference_frame)

        self.previous_difference_mask=self.difference_mask
        self.difference_mask=cv2.threshold(difference, .01, 1, cv2.THRESH_BINARY)[1]
        self.difference_mask=cv2.erode(self.difference_mask, np.ones((5,5)), iterations=2)
        self.difference_mask=cv2.dilate(self.difference_mask, np.ones((5,5)), iterations=5)
        self.inverse_difference_mask=1-self.difference_mask

        self.reference_frame=self.reference_frame*self.difference_mask+self.frame*self.inverse_difference_mask

        self.corrector*=self.difference_mask
        self.corrector+=.005*self.difference_mask
        self.corrector_mask=cv2.threshold(self.corrector, .8, 1, cv2.THRESH_BINARY)[1]
        self.reference_frame=self.corrector_mask*self.frame+(1-self.corrector_mask)*self.reference_frame

        if self.difference_mask.cumsum()[-1]>self.frame.shape[0]*self.frame.shape[1]*self.total_difference_threshold:
            self.reference_frame=self.reference_frame*self.previous_difference_mask+self.frame*(1-self.previous_difference_mask)
            self.difference_mask.fill(0)
            self.inverse_difference_mask.fill(1)
        return self.difference_mask


    def update_reference_frame(self):
        self.reference_frame=self.frame


if __name__=='__main__':
    cap=video.create_capture(0)
    ret=False
    while not ret:
        ret, frame=cap.read()
    frame=cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    bs=BackgroundSubtractor(frame)
    print "Press esc to exit; r to refresh reference frame"
    while True:
        frame=cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", frame)
        mask=bs.apply(frame)
        cv2.imshow("reference frame", bs.reference_frame)
        cv2.imshow("foreground", mask)
        k=cv2.waitKey(10)
        if k==1048690:
            bs.update_reference_frame()
        if k==1048603:
            cv2.destroyAllWindows()
            quit()
