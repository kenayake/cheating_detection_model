
def load_dt(data_dir, transforms):
    test_dataset = datasets.ImageFolder(
                os.path.join(data_dir, "test"), transform=self.transforms
            )
    
    test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

def evaluate(self, model, loader):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(
        all_labels, all_preds, average="macro", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, accuracy, precision, recall, cm, all_labels, all_preds
    

if __name__ == "__main__":
    pass