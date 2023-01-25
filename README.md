# sentinel_industry




This is code used to detect heavy industry sites, like coal power stations and steel melting plants, in satellite imagery. 

process_known_sources.ipynb filters the locations from the coal and steel plant tracker excel sheets and produces polygons.
Polygons are downloaded with the sentinel api (not part of this repo)

the train_* and try_* jupyter notebooks are for training different model architectures. Finally, only train_res50v2.ipynb is the one that is used further. 

filter_misclassified_cleanlab.ipynb filters out noisy training data, where images are mislabelled or cloud-contaminated. 

deploy_test.ipynb is for deploying the lower resolution model to a large region of land in northern India. 

the subfolder bingmaps/ contains files that replicate the above steps but for a second stage, higher resolution model (input of 1400x1400 instead of 120x120 pixels) 

