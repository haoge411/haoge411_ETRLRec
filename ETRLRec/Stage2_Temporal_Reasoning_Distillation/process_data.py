import sys

def _split_dataset(dis_dataset):
    try:
        total_size = len(dis_dataset)
        train_end = int(0.6 * total_size)  
        test_end = int(0.9 * total_size)
        
        all_indices = list(range(total_size))
        train_indices = all_indices[:train_end]
        test_indices = all_indices[train_end:test_end]
        eval_indices = all_indices[test_end:]

        train_dataset = dis_dataset.select(train_indices)
        test_dataset = dis_dataset.select(test_indices)
        eval_dataset = dis_dataset.select(eval_indices)
        return train_dataset, test_dataset, eval_dataset
    except Exception as e:
        print(f"Split dataset failed: {e}")
        sys.exit()
