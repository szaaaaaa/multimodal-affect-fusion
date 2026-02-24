# 数据说明（中文版）

## LSL
每位玩家会创建以下 LSL outlet：
- `IntData_<playerID>`：固定采样率的整数数据 outlet（采样率取决于服务器 FPS 设置）
- `FloatData_<playerID>`：固定采样率的浮点数据 outlet（采样率取决于服务器 FPS 设置）
- `MarkerData_<playerID>`：标记（marker）outlet

### 整数数据（integer data）
| channel | Label            | Format  | 说明 |
|--------|------------------|---------|------|
| 0      | tick             | Decimal | 样本（数据行）的 tick 编号，120Hz 采样 |
| 1      | round            | Decimal | 回合编号，120Hz 采样 |
| 2      | latency          | Decimal | 玩家延迟（毫秒） |
| 3      | isDucking        | Decimal | 玩家是否蹲下，1 表示是 |
| 4      | isJumping        | Decimal | 玩家是否跳跃，1 表示是 |
| 5      | health           | Decimal | 玩家生命值 |
| 6      | armor            | Decimal | 玩家护甲值 |
| 7      | score            | Decimal | 玩家得分 |
| 8      | money            | Decimal | 游戏内金钱 |
| 9      | slot0            | Decimal | 对应武器槽位的 CSWeaponID |
| 10     | slot1            | Decimal | 对应武器槽位的 CSWeaponID |
| 11     | slot2            | Decimal | 对应武器槽位的 CSWeaponID |
| 12     | slot3            | Decimal | 对应武器槽位的 CSWeaponID |
| 13     | slot4            | Decimal | 对应武器槽位的 CSWeaponID |
| 14     | slot5            | Decimal | 对应武器槽位的 CSWeaponID |
| 15     | slot6            | Decimal | 对应武器槽位的 CSWeaponID |
| 16     | slot7            | Decimal | 对应武器槽位的 CSWeaponID |
| 17     | slot8            | Decimal | 对应武器槽位的 CSWeaponID |
| 18     | slot9            | Decimal | 对应武器槽位的 CSWeaponID |
| 19     | slot10           | Decimal | 对应武器槽位的 CSWeaponID |
| 20     | slot11           | Decimal | 对应武器槽位的 CSWeaponID |
| 21     | equipped         | Decimal | 玩家当前装备武器的 CSWeaponID |
| 22     | magazine         | Decimal | 当前装备武器弹匣剩余子弹 |
| 23     | reserve          | Decimal | 当前装备武器备弹数量 |
| 24     | isReloading      | Decimal | 是否正在换弹，1 表示是 |
| 25     | bulletShots      | Decimal | 自上个采样点以来发射子弹数 |
| 26     | bulletHits       | Decimal | 命中玩家的子弹数 |
| 27     | hitGroup         | Decimal | 命中部位 `{generic,head,chest,belly,arm,legs}` |
| 28     | damage           | Decimal | 造成伤害值 |
| 29     | victimAlly       | Decimal | 受害友军玩家 ID |
| 30     | damageToAlly     | Decimal | 对友军造成的伤害 |
| 31     | attackerAlly     | Decimal | 友军攻击者玩家 ID |
| 32     | damageFromAlly   | Decimal | 来自友军的伤害 |
| 33     | victimEnemy      | Decimal | 受害敌军玩家 ID |
| 34     | damageToEnemy    | Decimal | 对敌军造成的伤害 |
| 35     | attackerEnemy    | Decimal | 敌军攻击者玩家 ID |
| 36     | damageFromEnemy  | Decimal | 来自敌军的伤害 |
| 37     | deathVictim      | Decimal | 死亡事件受害者 ID |
| 38     | deathAttacker    | Decimal | 死亡事件攻击者 ID |
| 39     | assistVictim     | Decimal | 助攻事件受害者 ID |
| 40     | assistAttacker   | Decimal | 助攻者 ID |
| 41     | inFOV1           | Decimal | 玩家 1 是否在视野内，1 表示是 |
| 42     | inFOV2           | Decimal | 玩家 2 是否在视野内，1 表示是 |
| 43     | inFOV3           | Decimal | 玩家 3 是否在视野内，1 表示是 |
| 44     | inFOV4           | Decimal | 玩家 4 是否在视野内，1 表示是 |
| 45     | aimTarget        | Decimal | 准星当前瞄准目标 |
| 46     | aimBodyPart      | Decimal | 瞄准部位 `{head,body,legs}` |
| --     | team1            | Decimal | 玩家 1 所属队伍 |
| --     | team2            | Decimal | 玩家 2 所属队伍 |
| --     | team3            | Decimal | 玩家 3 所属队伍 |
| --     | team4            | Decimal | 玩家 4 所属队伍 |

### 浮点数据（float data）
| channel | Label         | Format | 说明 |
|--------|---------------|--------|------|
| 0      | time          | Float  | 日志开始后的时间戳（秒） |
| 1      | positionX     | Float  | 玩家视点位置 X |
| 2      | positionY     | Float  | 玩家视点位置 Y |
| 3      | positionZ     | Float  | 玩家视点位置 Z |
| 4      | velocityX     | Float  | 玩家速度向量 X |
| 5      | velocityY     | Float  | 玩家速度向量 Y |
| 6      | velocityZ     | Float  | 玩家速度向量 Z |
| 7      | speed         | Float  | 玩家速度标量 |
| 8      | eyeVectorX    | Float  | 视线方向向量 X |
| 9      | eyeVectorY    | Float  | 视线方向向量 Y |
| 10     | eyeVectorZ    | Float  | 视线方向向量 Z |
| 11     | distance1     | Float  | 到玩家 1 的距离（未知时为 -1） |
| 12     | degressError1 | Float  | 准星命中玩家 1 的角度误差（未知时为 -1） |
| 13     | distanceError1| Float  | 准星到玩家 1 命中平面的距离误差（未知时为 -1） |
| 14     | distance2     | Float  | 到玩家 2 的距离（未知时为 -1） |
| 15     | degreesError2 | Float  | 准星命中玩家 2 的角度误差（未知时为 -1） |
| 16     | distanceError2| Float  | 准星到玩家 2 命中平面的距离误差（未知时为 -1） |
| 17     | distance3     | Float  | 到玩家 3 的距离（未知时为 -1） |
| 18     | degreesError3 | Float  | 准星命中玩家 3 的角度误差（未知时为 -1） |
| 19     | distanceError3| Float  | 准星到玩家 3 命中平面的距离误差（未知时为 -1） |
| 20     | distance4     | Float  | 到玩家 4 的距离（未知时为 -1） |
| 21     | degreesError4 | Float  | 准星命中玩家 4 的角度误差（未知时为 -1） |
| 22     | distanceError4| Float  | 准星到玩家 4 命中平面的距离误差（未知时为 -1） |
| --     | screenX1      | Float  | 玩家 1 在屏幕中的 X（不在视野内时为 NaN） |
| --     | screenY1      | Float  | 玩家 1 在屏幕中的 Y（不在视野内时为 NaN） |
| --     | screenX2      | Float  | 玩家 2 在屏幕中的 X（不在视野内时为 NaN） |
| --     | screenY2      | Float  | 玩家 2 在屏幕中的 Y（不在视野内时为 NaN） |
| --     | screenX3      | Float  | 玩家 3 在屏幕中的 X（不在视野内时为 NaN） |
| --     | screenY3      | Float  | 玩家 3 在屏幕中的 Y（不在视野内时为 NaN） |
| --     | screenX4      | Float  | 玩家 4 在屏幕中的 X（不在视野内时为 NaN） |
| --     | screenY4      | Float  | 玩家 4 在屏幕中的 Y（不在视野内时为 NaN） |

### 标记（markers）
每回合开始时会发送部分 marker 信息，格式为类 JSON：
- Players
- Weapons
- AimBodyParts
- HitGroups
