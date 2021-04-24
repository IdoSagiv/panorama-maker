import os
import shutil

import panorama_maker
import time


def main():
    videos = ['boat.mp4', 'iguazu.mp4']
    os.system('mkdir dump')
    for video in videos:
        exp_no_ext = video.split('.')[0]
        os.system(('mkdir ' + str(os.path.join('dump', '%s'))) % exp_no_ext)
        os.system(
            ('ffmpeg -i ' + str(os.path.join('videos', '%s ')) + str(os.path.join('dump', '%s', '%s%%03d.jpg'))) % (
                video, exp_no_ext, exp_no_ext))

        s = time.time()
        panorama_generator = panorama_maker.PanoramicVideoGenerator(
            os.path.join('dump', '%s') % exp_no_ext, exp_no_ext, 2100)
        panorama_generator.align_images(translation_only='boat' in video)
        panorama_generator.generate_panoramic_images(9)
        print(' time for %s: %.1f' % (exp_no_ext, time.time() - s))

        panorama_generator.save_panoramas_to_video()

    shutil.rmtree('dump')


if __name__ == '__main__':
    main()
