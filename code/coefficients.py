''' author: samtenka
    change: 2020-01-15
    create: 2019-06-18
    descrp: formulae for Taylor coefficients of loss in terms of diagram values
'''

sgd_vanilla_test = {
    0: '+     Dpln',
    1: '- T:1 Ipln',
    2: '+ T:2 Vpln' 
       '+ T:1 Vlvs / 2!',
    3: '- T:3 (4 Zpln + 2 Ypln)' 
       '- T:2 (1.5 Ylvs + Zlvs + Zalt)'
       '- T:1 Yerg / 3!',
}

sgd_vanilla_gauss_test = {
    0: '+     Dpln',
    1: '- T:1 Ipln',
    2: '+ T:2 Vpln' 
       '+ T:1 Vlvs / 2!',
    3: '- T:3 (4 Zpln + 2 Ypln)' 
       '- T:2 (1.5 Ylvs + Zlvs + Zalt)'
       '- T:1 (3 Ylvs - 2 Ypln) / 3!',
}

sde_test = {
    0: '+            Dpln                    ',
    1: '- ((T^1)/1!) Ipln                    ',
    2: '+ ((T^2)/2!) Vpln                    ' 
       '+ ((T^1)/1!) Vlvs / 2!               ',
    3: '- ((T^3)/3!) (4 Zpln + 2 Ypln)       ' 
       '- ((T^2)/2!) (1.5 Ylvs + Zlvs + Zalt)'
}

sgd_gauss_minus_sde_vanilla_test = {
    0: '(+     Dpln                     )-(+              Dpln                    )         ',
    1: '(- T:1 Ipln                     )-(- ((T^1)/(1!)) Ipln                    )         ',
    2: '(+ T:2 Vpln                     )-(+ ((T^2)/(2!)) Vpln                    ) +  ' 
       '(+ T:1 Vlvs / 2!                )-(+ ((T^1)/(1!)) Vlvs / 2!               )         ',
    3: '(- T:3 (4 Zpln + 2 Ypln)        )-(- ((T^3)/(3!)) (4 Zpln + 2 Ypln)       ) +' 
       '(- T:2 (1.5 Ylvs + Zlvs + Zalt) )-(- ((T^2)/(2!)) (1.5 Ylvs + Zlvs + Zalt)) +'
       '(- T:1 (3 Ylvs - 2 Ypln) / 3!   )                                           ',
}

sgd_minus_sde_vanilla_test = {
    0: '(+     Dpln                     )-(+              Dpln                    )         ',
    1: '(- T:1 Ipln                     )-(- ((T^1)/(1!)) Ipln                    )         ',
    2: '(+ T:2 Vpln                     )-(+ ((T^2)/(2!)) Vpln                    ) +  ' 
       '(+ T:1 Vlvs / 2!                )-(+ ((T^1)/(1!)) Vlvs / 2!               )         ',
    3: '(- T:3 (4 Zpln + 2 Ypln)        )-(- ((T^3)/(3!)) (4 Zpln + 2 Ypln)       ) +' 
       '(- T:2 (1.5 Ylvs + Zlvs + Zalt) )-(- ((T^2)/(2!)) (1.5 Ylvs + Zlvs + Zalt)) +'
       '(- T:1 Yerg / 3!                )                                           ',
}

sgd_vanilla_gen = {
    0: '-            0.0 ',
    1: '+ (T:1 / N) (  Iall -   Ipln)',
    2: '- (T:2 / N) (3 Vtwg +   Vlvs - 4 Vpln)' 
       '- (T:1 / N) (  Vall -   Vlvs) / 2!',
    3: '+ (T:3 / N) (4 Ztwg + 5 Zalt + 2 Zmid + 1 Zlvs - 12 Zpln)'
       '+ (T:3 / N) (4 Ytwg + 2 Ylvs - 6 Ypln)' 
       '+ (T:2 / N) (1 Yvee + 1.5 Ysli + 0.5 Ylvs - 3 Ylvs)'
       '+ (T:2 / N) (1 Ysli + 1 Yvee - 2 Ylvs)'
       '+ (T:2 / N) (1 Yvee + 1 Ysli - 2 Ytwg)'
       '+ (T:1 / N) (Yall - Yerg) / 3!',
}

gd_minus_sgd_vanilla_test = {
    0: '+                0.0 ',
    1: '-                0.0 ',
    2: '- ((T:2) / N) (Vtwg - Vpln)' 
}

sgd_linear_screw_renorm_z = {
    0: '+                0.0 ',
    1: '-                0.0 ',
    2: '+              T   /6',
    3: '-                0.0 ',
}

sgd_linear_screw_z = {
    0: '+                0.0 ',
    1: '-                0.0 ',
    2: '+                0.0 ',
    3: '- (T:2) Ylvs / 2!    ',
}

sgd_multi_test = {
    0: '+     Dpln',
    1: '- T:1 Ipln',
    2: '+ T (T-1/2) Vpln' 
       '+ (T T / N) (Vlvs-Vpln) / 2!' 
       '+ (T (T/N - 1)) (Vtwg-Vpln) / 2!'
}

sgd_multi_minus_vanilla_test = {
    0: '+     0.0 ',
    1: '-     0.0 ',
    2: '+ (N/T)^2 (T (T-1/2) Vpln         ) - N (N-1/2) Vpln         ' 
       '+ (N/T)^2 ((T T / N) (Vlvs-Vpln) / 2!    ) - (N N / N) (Vlvs-Vpln) / 2!    ' 
       '+ (N/T)^2 ((T (T/N - 1)) (Vtwg-Vpln) / 2!) - (N (N/N - 1)) (Vtwg-Vpln) / 2!'
}

