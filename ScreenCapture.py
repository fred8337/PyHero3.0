import cv2
import time

import numpy as np
import win32gui, win32ui, win32con, win32api


def grab_screen(region=None, color=cv2.COLOR_BGRA2RGB):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, color)

if __name__ == "__main__":
    start_time = time.time()
    number_of_frames = 0;
    while True:
        img = grab_screen(region=(0, 0, 800, 600))
        # cv2.imshow("Lol", img)
        # cv2.waitKey(25)
        number_of_frames+=1
        if number_of_frames%15==0:
            seconds = time.time()-start_time
            fps = number_of_frames / seconds
            print("Estimated frames per second : {0}".format(fps))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break