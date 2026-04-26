from src.models.ViT import ViT
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='run seed')
    args = parser.parse_args()
    model = ViT(input_size=224,
                patch_size=16,
                num_classes=10,
                dim=768,
                depth=12,
                heads=12,
                mlp_dim=3072,
                dropout=0.1)

if __name__ == "__main__":
    main()