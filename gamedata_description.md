# Notes on data

## LSL
The following outlets are created for each player:
* IntData\_\<playerID\>: A constant rate integer data outlet (rate depends on server fps setting)
* FloatData\_\<playerID\>: A constant rate float data outlet (rate depends on server fps setting)
* MarkerData\_\<playerID\>: A marker outlet

### integer data
|channel| Label              | Format | Notes                                   |
|-------|--------------------|--------|-----------------------------------------|
|0      |tick                |Decimal |Tick number of sample (data row). Sampled at 120Hz |
|1      |round                |Decimal |Tick number of sample (data row). Sampled at 120Hz |
|2      |latency             |Decimal |Player latency in miliseconds|
|3      |isDucking           |Decimal |1 if player ducking |
|4      |isJumping           |Decimal |1 if player jumping |
|5      |health              |Decimal |Health of player |
|6      |armor               |Decimal |Armor of player |
|7      |score               |Decimal |Score of player |
|8      |money               |Decimal |In game money of player |
|9      |slot0               |Decimal |CSWeaponID of given slot|
|10     |slot1               |Decimal |CSWeaponID of given slot|
|11     |slot2               |Decimal |CSWeaponID of given slot|
|12     |slot3               |Decimal |CSWeaponID of given slot|
|13     |slot4               |Decimal |CSWeaponID of given slot|
|14     |slot5               |Decimal |CSWeaponID of given slot|
|15     |slot6               |Decimal |CSWeaponID of given slot|
|16     |slot7               |Decimal |CSWeaponID of given slot|
|17     |slot8               |Decimal |CSWeaponID of given slot|
|18     |slot9               |Decimal |CSWeaponID of given slot|
|19     |slot10              |Decimal |CSWeaponID of given slot|
|20     |slot11              |Decimal |CSWeaponID of given slot|
|21     |equipped            |Decimal |CSWeaponID equipped by player |
|22     |magazine            |Decimal |Bullets left in equipped weapon magazine |
|23     |reserve             |Decimal |Bullets in reserve of equipped weapon |
|24     |isReloading         |Decimal |1 if player is reloading weapon |
|25     |bulletShots         |Decimal |Number of bullets shot since last sample |
|26     |bulletHits          |Decimal |Number of bullets that hit a player|
|27     |hitGroup            |Decimal |{generic,head,chest,belly,arm,legs} |
|28     |damage              |Decimal |Damage dealt |
|29     |victimAlly          |Decimal |Ally player ID victim |
|30     |damageToAlly        |Decimal |Damage dealt to ally |
|31     |attackerAlly        |Decimal |Ally player ID attacker|
|32     |damageFromAlly      |Decimal |Damage received from ally |
|33     |victimEnemy         |Decimal |Enemy player ID victim |
|34     |damageToEnemy       |Decimal |Damage dealt to enemy |
|35     |attackerEnemy       |Decimal |Enemy player ID attacker |
|36     |damageFromEnemy     |Decimal |Damage received from enemy |
|37     |deathVictim         |Decimal |Victim player ID |
|38     |deathAttacker       |Decimal |Attacker player ID |
|39     |assistVictim        |Decimal |Victim player ID|
|40     |assistAttacker      |Decimal |Attacker (assister) player ID|
|41     |inFOV1              |Decimal |1 if player 1 is in field of view|
|42     |inFOV2              |Decimal |1 if player 2 is in field of view |
|43     |inFOV3              |Decimal |1 if player 3 is in field of view |
|44     |inFOV4              |Decimal |1 if player 4 is in field of view |
|45     |aimTarget           |Decimal |Crosshair that player aimed at |
|46     |aimBodyPart         |Decimal |{head,body,legs} |
|--     |team1         		 |Decimal |team of player 1 |
|--     |team2         		 |Decimal |team of player 2 |
|--     |team3         		 |Decimal |team of player 3 |
|--     |team4         		 |Decimal |team of player 4 |

### float data
|channel| Label              | Format | Notes                                   |
|-------|--------------------|--------|-----------------------------------------|
|0      |time                |Float   |Timestamp since start of logs in seconds |
|1      |positionX           |Float   |Player eye position x |
|2      |positionY           |Float   |Player eye position y|
|3      |positionZ           |Float   |Player eye position z |
|4      |velocityX           |Float   |Player velocity x |
|5      |velocityY           |Float   |Player velocity y |
|6      |velocityZ           |Float   |Player velocity z |
|7      |speed               |Float   |Player speed |
|8      |eyeVectorX          |Float   |Player eye direction of view x |
|9      |eyeVectorY          |Float   |Player eye direction of view y|
|10     |eyeVectorZ          |Float   |Player eye direction of view z |
|11     |distance1           |Float   |Distance to player 1 (-1 if unknown) |
|12     |degressError1       |Float   |Degrees error of crosshair to hit player 1 (-1 if unknown) |
|13     |distanceError1      |Float   |Distance of crosshair from hitplane of player 1 (-1 if unknown) |
|14     |distance2           |Float   |Distance to player 2 (-1 if unknown) |
|15     |degreesError2       |Float   |Degrees error of crosshair to hit player 2 (-1 if unknown) |
|16     |distanceError2      |Float   |Distance of crosshair from hitplane of player 2 (-1 if unknown) |
|17     |distance3           |Float   |Distance to player 3 (-1 if unknown) |
|18     |degreesError3       |Float   |Degrees error of crosshair to hit player 3 (-1 if unknown) |
|19     |distanceError3      |Float   |Distance of crosshair from hitplane of player 3 (-1 if unknown) |
|20     |distance4           |Float   |Distance to player 4 (-1 if unknown) |
|21     |degreesError4       |Float   |Degrees error of crosshair to hit player 4 (-1 if unknown) |
|22     |distanceError4      |Float   |Distance of crosshair from hitplane of player 4 (-1 if unknown) |
|--     |screenX1            |Float   |X Position of player 1 on the screen (nan if not infov)|
|--     |screenY1            |Float   |Y Position of player 1 on the screen (nan if not infov)|
|--     |screenX2            |Float   |X Position of player 2 on the screen (nan if not infov)|
|--     |screenY2            |Float   |Y Position of player 2 on the screen (nan if not infov)|
|--     |screenX3            |Float   |X Position of player 3 on the screen (nan if not infov)|
|--     |screenY3            |Float   |Y Position of player 3 on the screen (nan if not infov)|
|--     |screenX4            |Float   |X Position of player 4 on the screen (nan if not infov)|
|--     |screenY4            |Float   |Y Position of player 4 on the screen (nan if not infov)|


### markers
Some marker information is sent at every round start in a JSON-like format:
* Players
* Weapons
* AimBodyParts
* HitGroups
