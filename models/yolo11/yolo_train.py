import scipy as sp
from ultralytics import YOLO
from roboflow import Roboflow
from wakepy import keep

def initialize_roboflow():
    rf = Roboflow(api_key="m2wBNOl1tzsXZBfl8Lwv")
    project = rf.workspace("skripsi-gfit0").project("cctv-detection-2-clpgw")
    dataset = project.version(5).download("yolov11", location="dataset/yolo/cctv-detection-alt-5")
    return dataset

def train_model(dataset):
    model = YOLO('yolo11s.pt')
    with keep.running():
        results = model.train(
            data=f"{dataset.location}/data.yaml",
            epochs=300,
            imgsz=640,
            batch=16,
            name='yolo11s_dataset_alt_v5',
            loggers=["tensorboard"],
        )

        metrics = model.val()
        
        test_metrics = model.val(split="test")

        model.export()

def main():
    dataset = initialize_roboflow()
    train_model(dataset)

if __name__ == "__main__":
    main()