import os
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write

from mel2wav.modules import Generator

MAX_WAV_VALUE = 32768.0

def main(args):
    checkpoint = torch.load(args.checkpoint_path + '/best_netG.pt')
    
    model = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).cuda()
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        for melpath in tqdm.tqdm(glob.glob(os.path.join(args.input_folder, '*.npy'))):
            print(melpath)
            mel = torch.from_numpy(np.load(melpath).T)
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.cuda()

            audio = model.forward(mel)
            audio = audio.cpu().detach().numpy()

            out_path = melpath.replace('.npy', '.wav')
            write(out_path, audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help='path of checkpoint pt file for evaluation')
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help='directory of mel-spectrograms to invert into raw audio.')
    
    parser.add_argument('-n_mel_channels', type=int, default=80)
    parser.add_argument('-ngf', type=int, default=32)
    parser.add_argument('-n_residual_layers', type=int, default=3)

    args = parser.parse_args()

    main(args)