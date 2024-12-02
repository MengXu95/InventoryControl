import sys
import CCGP_niching.GPFC as CCGPmain_niching
import MTGP_niching.GPFC as GPmain_niching
import MTGP_niching_rental.GPFC as GPmain_niching_rental
import DRL.DRL_main as DRL_main
import MTGP_niching.testRuleMTGP as testRuleMTGP
import CCGP_niching.testRuleCCGP as testRuleCCGP
import DRL.testRuleDRL as testRuleDRL
import sSPolicy.trainRulesSPolicy as trainRulesSPolicy

sys.path

if __name__ == '__main__':
    dataset_name = str(sys.argv[1])
    seed = int(sys.argv[2])
    algo = str(sys.argv[3])

    if algo == 'CCGP_niching':
        print('----------CCGP_niching----------')
        CCGPmain_niching.main(dataset_name, seed)
    elif algo == 'MTGP_niching':
        print('----------MTGP_niching----------')
        GPmain_niching.main(dataset_name, seed)
    elif algo == 'MTGP_niching_rental':
        print('----------MTGP_niching_rental----------')
        GPmain_niching_rental.main(dataset_name, seed)
    elif algo == 'DRL':
        print('----------DRL----------')
        DRL_main.main(dataset_name, seed)
    elif algo == 'testRuleMTGP':
        print('----------testRuleMTGP----------')
        testRuleMTGP.main(dataset_name, seed)
    elif algo == 'testRuleCCGP':
        print('----------testRuleCCGP----------')
        testRuleCCGP.main(dataset_name, seed)
    elif algo == 'testRuleDRL':
        print('----------testRuleDRL----------')
        testRuleDRL.main(dataset_name, seed)
    elif algo == 'trainRulesSPolicy':
        print('----------trainAndTestRulesSPolicy----------')
        trainRulesSPolicy.main(dataset_name, seed)
