Celeb-DF Dataset
The Celeb-DF (Celebrity DeepFake) Dataset is a large collection of real and deepfake videos featuring celebrities. It is designed for deepfake detection research, helping to develop models that distinguish between authentic and manipulated content.

🔗 Dataset Link: [Celeb-DF Official Website](https://paperswithcode.com/dataset/celeb-df)

📂 Dataset Structure
Train Data:
Contains 590 real videos and 5,639 deepfake videos in .mp4 format.

Test Data:
Contains a subset of real and fake videos for evaluation.

📝 MetaData of the Dataset
Column	Description
filename	The filename of the video.
label	Whether the video is REAL (0) or FAKE (1).
original	If the video is FAKE, this column contains the name of the original video.
split	Indicates whether the video belongs to "train" or "test".
🔽 Downloading the Dataset
The dataset is not publicly available for direct download. You need to request access from the official website and extract the dataset into your working directory.
