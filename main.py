import sys
import CCGP_niching.GPFC as CCGPmain_niching
import MTGP_niching.GPFC as GPmain_niching
import MTGP_niching_rental.GPFC as GPmain_niching_rental
import CCGP_niching_rental.GPFC as CCGPmain_niching_rental
import MTGP_niching_rental_original.GPFC as GPmain_niching_rental_original
import CCGP_niching_rental_original.GPFC as CCGPmain_niching_rental_original
import PSO_rental.PSO_main as PSO_main_rental
import DRL.DRL_main as DRL_main
import MTGP_niching.testRuleMTGP as testRuleMTGP
import CCGP_niching.testRuleCCGP as testRuleCCGP
import DRL.testRuleDRL as testRuleDRL
import sSPolicy.trainRulesSPolicy as trainRulesSPolicy
import S2Demo as S2Demo

sys.path

if __name__ == '__main__':
    dataset_name = str(sys.argv[1])
    seed = int(sys.argv[2])
    algo = str(sys.argv[3])


    if algo == 'CCGP_niching':
        print('----------CCGP_niching----------')
        CCGPmain_niching.main(dataset_name, seed)
    elif algo == 'CCGP_niching_rental':
        print('----------CCGP_niching_rental----------')
        CCGPmain_niching_rental.main(dataset_name, seed)
    elif algo == 'CCGP_niching_rental_2':
        for i in range(2):
            print('----------CCGP_niching_rental_'+str(i)+'_----------')
            seed = i
            CCGPmain_niching_rental.main(dataset_name, seed)
    elif algo == 'MTGP_niching':
        print('----------MTGP_niching----------')
        GPmain_niching.main(dataset_name, seed)
    elif algo == 'MTGP_niching_rental':
        print('----------MTGP_niching_rental----------')
        GPmain_niching_rental.main(dataset_name, seed)
    elif algo == 'MTGP_niching_rental_2':
        for i in range(3):
            print('----------MTGP_niching_rental_'+str(i)+'_----------')
            seed = i
            GPmain_niching_rental.main(dataset_name, seed)
    elif algo == 'MTGP_niching_rental_original_3':
        for i in range(3):
            print('----------MTGP_niching_rental_original_'+str(i)+'_----------')
            seed = i
            GPmain_niching_rental_original.main(dataset_name, seed)
    elif algo == 'CCGP_niching_rental_original_3':
        for i in range(3):
            print('----------CCGP_niching_rental_original_'+str(i)+'_----------')
            seed = i
            CCGPmain_niching_rental_original.main(dataset_name, seed)
    elif algo == 'PSO_rental':
        print('----------PSO_rental----------')
        PSO_main_rental.main(dataset_name, seed)
    elif algo == 'PSO_rental_3':
        for i in range(3):
            print('----------PSO_rental_'+str(i)+'_----------')
            seed = i
            PSO_main_rental.main(dataset_name, seed)
    elif algo == 'DRL':
        print('----------DRL----------')
        DRL_main.main(dataset_name, seed)
    elif algo == 'testRuleMTGP':
        print('----------testRuleMTGP----------')
        testRuleMTGP.main(dataset_name, seed)
    elif algo == 'S2Demo':
        print('----------S2Demo----------')
        S2Demo.main(dataset_name, seed)
    elif algo == 'testRuleCCGP':
        print('----------testRuleCCGP----------')
        testRuleCCGP.main(dataset_name, seed)
    elif algo == 'testRuleDRL':
        print('----------testRuleDRL----------')
        testRuleDRL.main(dataset_name, seed)
    elif algo == 'trainRulesSPolicy':
        print('----------trainAndTestRulesSPolicy----------')
        trainRulesSPolicy.main(dataset_name, seed)


    # for batch run
    # dataset_name = "mN2h_1_5b2"
    # for i in range(3):
    #     print('----------MTGP_niching_rental_original_' + str(i) + '_----------')
    #     seed = i
    #     GPmain_niching_rental_original.main(dataset_name, seed)
    #
    # for i in range(3):
    #     print('----------CCGP_niching_rental_original_' + str(i) + '_----------')
    #     seed = i
    #     CCGPmain_niching_rental_original.main(dataset_name, seed)
    #
    # dataset_name = "lN2h_1_5b2"
    # for i in range(3):
    #     print('----------MTGP_niching_rental_original_' + str(i) + '_----------')
    #     seed = i
    #     GPmain_niching_rental_original.main(dataset_name, seed)
    #
    # for i in range(3):
    #     print('----------CCGP_niching_rental_original_' + str(i) + '_----------')
    #     seed = i
    #     CCGPmain_niching_rental_original.main(dataset_name, seed)
    #
    # for i in range(3):
    #     print('----------PSO_rental_' + str(i) + '_----------')
    #     seed = i
    #     PSO_main_rental.main(dataset_name, seed)