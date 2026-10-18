IFAN drop zone
==============

Put images here, or whole folders of images, then double-click run_infer.bat
in the folder above.

Results are written back into this folder, in a "deblurred" subfolder and with
a "_db" suffix on every file, so they can never be confused with the originals:

    input\
      holiday\
        IMG_0001.jpg
        IMG_0002.jpg
        evening\
          IMG_0003.jpg
      one-off-shot.png

Running run_infer.bat with no arguments produces:

    input\
      deblurred\
        holiday\
          IMG_0001_db.jpg
          IMG_0002_db.jpg
          evening\
            IMG_0003_db.jpg
        one-off-shot_db.jpg
      holiday\...            (the originals, untouched)
      one-off-shot.png

Notes
-----

* No options are needed.  Large images are tiled automatically and results are
  always produced at full resolution - nothing is downscaled.
* Subfolders are always processed, and the folder structure is mirrored inside
  "deblurred".
* Only these extensions are picked up: jpg jpeg png bmp tif tiff webp
* Output size is trimmed to a multiple of 8 pixels (at most 7 px smaller), which
  the network requires.
* Images over 100 megapixels are skipped with a message naming the limit, rather
  than being shrunk.  Everything else in the folder is still processed.
* Re-running is safe.  Everything under "deblurred", and anything already ending
  in "_db", is skipped, so the folder does not grow on each run.
* Dragging a folder onto run_infer.bat also works and puts the results next to
  that folder instead:
      D:\photos\IMG_0001.jpg  ->  D:\photos\deblurred\IMG_0001_db.jpg
* To keep results out of this folder entirely, run instead:
      run_infer.bat --dest output
  which puts them in the "output" folder above.
* This folder is safe to empty or delete at any time.
