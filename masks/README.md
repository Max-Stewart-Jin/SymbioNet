# Zone maps

`houston2013_cloud_shadow.npz` marks the cloud shadow in the Houston2013 hyperspectral image. The eastern part of
the image lies in the shadow of a cloud; the LiDAR raster is not affected.

| array | content |
|---|---|
| `zone` | (349, 1905) int8: 2 = shadow core, 1 = uncertain (within 4 pixels of the shadow boundary, partly shaded areas, small shadow patches), 0 = outside, -1 = no-data pixel |
| `split` | the split file the counts below were made for (`houston2013_geodeltassm.npz`) |
| `test_counts`, `train_counts` | test / training pixels outside, uncertain and in the shadow core (13115 / 360 / 1396 and 140 / 3 / 15) |

How it was made: a gradient-boosting classifier on the log brightness of 12 spectral segments of every pixel,
trained on hand-marked rectangles (four inside the shadow, three outside), applied to the whole image, smoothed
with a 7 x 7 median filter; patches under 1500 pixels were removed and small holes filled. The pixels within 4
pixels of the boundary are the uncertain zone.

`python formal.py test tests1` scores every tested model inside and outside the shadow core on the clean test
pixels (see `symbionet/testing.py`, `zone_scores`).
