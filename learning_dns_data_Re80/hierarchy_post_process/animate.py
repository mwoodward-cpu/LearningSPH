import os

def save():
    os.system("ffmpeg -framerate 16 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p u_over_t.mp4")


save()