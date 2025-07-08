import argparse

from train_models.train_domain_adaptation import train_model as train_dann
from train_models.train_base import train_model as train_base


general_path = '../../../../../tudelft.net/staff-umbrella/MScThesisJLuu/data/'

# oai_files = [
#     "../OAI_part00.h5",
#     "../OAI_part01.h5",
#     "../OAI_part02.h5",
#     "../OAI_part03.h5",
#     "../OAI_part04.h5",
#     "../OAI_part05.h5",
# ]
#
# check_files = [
#     "../CHECK_part00.h5",
#     "../CHECK_part01.h5"
# ]

oai_files = [
    general_path + "OAI_part00.h5",
    general_path + "OAI_part01.h5",
    general_path + "OAI_part02.h5",
    general_path + "OAI_part03.h5",
    general_path + "OAI_part04.h5",
    general_path + "OAI_part05.h5",
]

check_files = [
    general_path + "CHECK_part00.h5",
    general_path + "CHECK_part01.h5"
]


def main():
    parser = argparse.ArgumentParser(description="Train Hip X-ray OA classifier")

    parser.add_argument(
        "--type",
        type=str,
        default="base",
        choices=["base", "mmd", "dann"],
    )

    parser.add_argument(
        "--source_dataset",
        type=str,
        default="oai",
        choices=["oai", "check"],
    )

    parser.add_argument(
        "--filename",
        type=str,
        default='some_classifier',
        help="name of file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--source_ratio",
        type=float,
        default=0.5,
        help="Ratio default is 0.5"
    )

    parser.add_argument(
        "--target_ratio",
        type=float,
        default=0.5,
        help="Ratio default is 0.5"
    )

    parser.add_argument(
        "--source_majority_class",
        type=int,
        default=0,
        choices=[0, 1],
    )

    parser.add_argument(
        "--target_majority_class",
        type=int,
        default=0,
        choices=[0, 1],
    )

    parser.add_argument(
        "--labda",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--same_size",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if args.type == "base":
        if args.source_dataset == "oai":
            train_base(oai_files, check_files, args.filename,
                       num_epochs=args.epochs, learning_rate=args.lr, source_majority_class=args.source_majority_class, target_majority_class=args.target_majority_class)
        else:
            train_base(check_files, oai_files, args.filename,
                       num_epochs=args.epochs, learning_rate=args.lr, source_majority_class=args.source_majority_class, target_majority_class=args.target_majority_class)
    else:
        if args.source_dataset == "oai":
            train_dann(oai_files, check_files, filename=args.filename, num_epochs=args.epochs, lr=args.lr, adaptation_type=args.type, source_majority_class=args.source_majority_class, target_majority_class=args.target_majority_class, source_ratio=args.source_ratio, target_ratio=args.target_ratio, same_size=args.same_size)
        else:
            train_dann(check_files, oai_files, filename=args.filename, num_epochs=args.epochs, lr=args.lr, adaptation_type=args.type, source_majority_class=args.source_majority_class, target_majority_class=args.target_majority_class, source_ratio=args.source_ratio, target_ratio=args.target_ratio, same_size=args.same_size)


if __name__ == "__main__":
    main()