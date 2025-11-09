### Policy Iteration

```
$ python policy_iteration.py
Converged in 1.09 seconds

Optimal policy table:
0000000002222222222222
0000200002222222222222
0002000002222222222222
0020000002222222222222
0000000000022222222222
0000000000022222222222
0000000000022222222222
0000000000022222222222
0000000000022222222222
0000000000002222222222
0000000000002222222222
0000000000002222222222
0000000000002222222222
0000000000000022222222
0000000000000022222222
0000000000000022222222
0000000000000022222222
0000000000002222222222
0000000000002222222222
0000000000002222222222
0000000000002222222222
0000000000002222220000

Average return: -111.3224
```

### Value Iteration

```
$ python value_iteration.py
Converged in 0.25 seconds

Optimal policy table:
0000000000222222222222
0000200000222222222222
0002000000222222222222
0020000000222222222222
0000000000022222222222
0000000000022222222222
0000000000022222222222
0000000000022222222222
0000000000022222222222
0000000000002222222222
0000000000002222222222
0000000000002222222222
0000000000002222222222
0000000000000222222222
0000000000000222222222
0000000000000222222222
0000000000000222222222
0000000000002222222222
0000000000002222222222
0000000000002222222222
0000000000002222222222
0000000000002222220000

Average return: -111.5254
```

### Monte Carlo

A custom training reward is used that encourages maximization of total energy
(potential plus kinetic).
Also, epsilon decay is used.
Training is much slower (tens of minutes) than dynamic programming methods.

```
$ python monte_carlo.py
Episode 1, training return: -3789.5, testing return: -5000.0, epsilon: 0.22
Episode 101, training return: -364.1, testing return: -840.0, epsilon: 0.22
Episode 201, training return: -98.4, testing return: -178.0, epsilon: 0.22
Episode 301, training return: -185.4, testing return: -443.0, epsilon: 0.21
Episode 401, training return: -124.4, testing return: -253.0, epsilon: 0.21
Episode 501, training return: -191.5, testing return: -337.0, epsilon: 0.21
Episode 601, training return: -238.5, testing return: -471.0, epsilon: 0.21
Episode 701, training return: -144.3, testing return: -250.0, epsilon: 0.21
Episode 801, training return: -159.4, testing return: -265.0, epsilon: 0.20
Episode 901, training return: -122.9, testing return: -257.0, epsilon: 0.20
Episode 1001, training return: -91.5, testing return: -177.0, epsilon: 0.20
Episode 1101, training return: -227.8, testing return: -462.0, epsilon: 0.20
Episode 1201, training return: -96.4, testing return: -179.0, epsilon: 0.20
Episode 1301, training return: -236.1, testing return: -462.0, epsilon: 0.19
Episode 1401, training return: -247.6, testing return: -516.0, epsilon: 0.19
Episode 1501, training return: -210.7, testing return: -367.0, epsilon: 0.19
Episode 1601, training return: -145.1, testing return: -283.0, epsilon: 0.19
Episode 1701, training return: -123.1, testing return: -247.0, epsilon: 0.19
Episode 1801, training return: -154.8, testing return: -358.0, epsilon: 0.18
Episode 1901, training return: -110.0, testing return: -189.0, epsilon: 0.18
Episode 2001, training return: -130.3, testing return: -272.0, epsilon: 0.18
Episode 2101, training return: -330.7, testing return: -437.0, epsilon: 0.18
Episode 2201, training return: -160.9, testing return: -336.0, epsilon: 0.18
Episode 2301, training return: -156.1, testing return: -342.0, epsilon: 0.17
Episode 2401, training return: -285.7, testing return: -580.0, epsilon: 0.17
Episode 2501, training return: -133.5, testing return: -267.0, epsilon: 0.17
Episode 2601, training return: -90.9, testing return: -170.0, epsilon: 0.17
Episode 2701, training return: -93.8, testing return: -178.0, epsilon: 0.17
Episode 2801, training return: -147.3, testing return: -273.0, epsilon: 0.17
Episode 2901, training return: -203.1, testing return: -330.0, epsilon: 0.16
Episode 3001, training return: -92.2, testing return: -170.0, epsilon: 0.16
Episode 3101, training return: -95.9, testing return: -176.0, epsilon: 0.16
Episode 3201, training return: -253.3, testing return: -413.0, epsilon: 0.16
Episode 3301, training return: -84.7, testing return: -162.0, epsilon: 0.16
Episode 3401, training return: -83.0, testing return: -166.0, epsilon: 0.16
Episode 3501, training return: -92.0, testing return: -177.0, epsilon: 0.16
Episode 3601, training return: -88.0, testing return: -171.0, epsilon: 0.15
Episode 3701, training return: -132.8, testing return: -259.0, epsilon: 0.15
Episode 3801, training return: -158.2, testing return: -346.0, epsilon: 0.15
Episode 3901, training return: -172.6, testing return: -312.0, epsilon: 0.15
Episode 4001, training return: -169.8, testing return: -291.0, epsilon: 0.15
Episode 4101, training return: -93.9, testing return: -173.0, epsilon: 0.15
Episode 4201, training return: -304.3, testing return: -453.0, epsilon: 0.14
Episode 4301, training return: -116.4, testing return: -257.0, epsilon: 0.14
Episode 4401, training return: -139.1, testing return: -218.0, epsilon: 0.14
Episode 4501, training return: -124.9, testing return: -262.0, epsilon: 0.14
Episode 4601, training return: -157.5, testing return: -251.0, epsilon: 0.14
Episode 4701, training return: -93.1, testing return: -184.0, epsilon: 0.14
Episode 4801, training return: -116.5, testing return: -252.0, epsilon: 0.14
Episode 4901, training return: -87.0, testing return: -171.0, epsilon: 0.13
Episode 5001, training return: -121.4, testing return: -251.0, epsilon: 0.13
Episode 5101, training return: -140.0, testing return: -298.0, epsilon: 0.13
Episode 5201, training return: -195.4, testing return: -382.0, epsilon: 0.13
Episode 5301, training return: -205.3, testing return: -417.0, epsilon: 0.13
Episode 5401, training return: -197.2, testing return: -306.0, epsilon: 0.13
Episode 5501, training return: -220.9, testing return: -370.0, epsilon: 0.13
Episode 5601, training return: -125.0, testing return: -258.0, epsilon: 0.13
Episode 5701, training return: -99.5, testing return: -184.0, epsilon: 0.12
Episode 5801, training return: -156.1, testing return: -262.0, epsilon: 0.12
Episode 5901, training return: -94.8, testing return: -176.0, epsilon: 0.12
Episode 6001, training return: -119.1, testing return: -249.0, epsilon: 0.12
Episode 6101, training return: -85.8, testing return: -171.0, epsilon: 0.12
Episode 6201, training return: -97.6, testing return: -183.0, epsilon: 0.12
Episode 6301, training return: -210.7, testing return: -325.0, epsilon: 0.12
Episode 6401, training return: -126.0, testing return: -254.0, epsilon: 0.12
Episode 6501, training return: -202.5, testing return: -310.0, epsilon: 0.11
Episode 6601, training return: -226.6, testing return: -487.0, epsilon: 0.11
Episode 6701, training return: -88.5, testing return: -172.0, epsilon: 0.11
Episode 6801, training return: -88.0, testing return: -177.0, epsilon: 0.11
Episode 6901, training return: -158.7, testing return: -346.0, epsilon: 0.11
Episode 7001, training return: -105.0, testing return: -188.0, epsilon: 0.11
Episode 7101, training return: -123.6, testing return: -270.0, epsilon: 0.11
Episode 7201, training return: -209.0, testing return: -311.0, epsilon: 0.11
Episode 7301, training return: -125.8, testing return: -252.0, epsilon: 0.11
Episode 7401, training return: -86.8, testing return: -170.0, epsilon: 0.10
Episode 7501, training return: -146.4, testing return: -268.0, epsilon: 0.10
Episode 7601, training return: -85.6, testing return: -177.0, epsilon: 0.10
Episode 7701, training return: -87.4, testing return: -176.0, epsilon: 0.10
Episode 7801, training return: -88.4, testing return: -178.0, epsilon: 0.10
Episode 7901, training return: -126.7, testing return: -205.0, epsilon: 0.10
Episode 8001, training return: -116.5, testing return: -201.0, epsilon: 0.10
Episode 8101, training return: -117.4, testing return: -199.0, epsilon: 0.10
Episode 8201, training return: -87.0, testing return: -167.0, epsilon: 0.10
Episode 8301, training return: -222.6, testing return: -366.0, epsilon: 0.10
Episode 8401, training return: -85.4, testing return: -169.0, epsilon: 0.09
Episode 8501, training return: -121.1, testing return: -263.0, epsilon: 0.09
Episode 8601, training return: -89.5, testing return: -170.0, epsilon: 0.09
Episode 8701, training return: -88.3, testing return: -165.0, epsilon: 0.09
Episode 8801, training return: -94.6, testing return: -171.0, epsilon: 0.09
Episode 8901, training return: -121.1, testing return: -265.0, epsilon: 0.09
Episode 9001, training return: -93.0, testing return: -178.0, epsilon: 0.09
Episode 9101, training return: -84.2, testing return: -169.0, epsilon: 0.09
Episode 9201, training return: -155.1, testing return: -288.0, epsilon: 0.09
Episode 9301, training return: -91.9, testing return: -187.0, epsilon: 0.09
Episode 9401, training return: -154.1, testing return: -266.0, epsilon: 0.09
Episode 9501, training return: -87.7, testing return: -163.0, epsilon: 0.09
Episode 9601, training return: -139.9, testing return: -218.0, epsilon: 0.08
Episode 9701, training return: -92.5, testing return: -175.0, epsilon: 0.08
Episode 9801, training return: -90.9, testing return: -172.0, epsilon: 0.08
Episode 9901, training return: -100.4, testing return: -178.0, epsilon: 0.08

Optimal policy table:
00001012112020000000
00121221211022200000
00112222200110120000
01221120111112012000
00121102102002220000
01121200001222221100
01102111001201121000
01222121002022222110
01221001000020222210
00221011000012220210
00220120220002021200
00111100102000022200
00022210010011022100
00002212120020112000
00000212210010221000
00000221121212200000
00000012122222200000
00000001211121200000
00000000112210000000
00000000020202000000

Average return: -180.2
```
