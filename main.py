import sys
import MTGP.GPFC as GPmain
import CCGP.GPFC as CCGPmain
import MTGP_new_terminals.GPFC as GPmain_new_terminals
import MTGP_niching.GPFC as GPmain_niching

sys.path

if __name__ == '__main__':
    dataset_name = str(sys.argv[1])
    seed = int(sys.argv[2])
    algo = str(sys.argv[3])

    if algo == 'MTGP':
        print('----------MTGP----------')
        GPmain.main(dataset_name, seed)
    elif algo == 'CCGP':
        print('----------CCGP----------')
        CCGPmain.main(dataset_name, seed)
    elif algo == 'MTGP_new_terminals':
        print('----------MTGP_new_terminals----------')
        GPmain_new_terminals.main(dataset_name, seed)
    elif algo == 'MTGP_niching':
        print('----------MTGP_niching----------')
        GPmain_niching.main(dataset_name, seed)