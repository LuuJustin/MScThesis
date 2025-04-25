from train_resnet18 import train_model


general_path = '../../../../tudelft.net/staff-umbrella/MScThesisJLuu/data'


oai_files = [
    general_path + "/OAI_part00.h5", general_path + "/OAI_part01.h5", general_path + "/OAI_part02.h5",
    general_path + "/OAI_part03.h5", general_path + "/OAI_part04.h5", general_path + "/OAI_part05.h5",
]

check_files = [
    general_path + "/CHECK_part00.h5", general_path + "/CHECK_part01.h5"
]

trained_model = train_model(oai_files, check_files, num_epochs=100, lr=1e-4)
