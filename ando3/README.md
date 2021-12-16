# 安藤さん卒研実験3


## Configurations(hyper parameters)

実験2と同様に、小さいアンカースケールを削除

| Parameter              |Value                   |
|------------------------|------------------------|
|**RPN_ANCHOR_RATIOS**   |[0.5, 1, 2]             |
|**RPN_ANCHOR_SCALES**   |(128, 256, 512) |
|**RPN_ANCHOR_STRIDE**   |1                       |

## Dataset
ピクセル面積600以下のフィラメントを教師データから除外

|--------------|------------------|
| train        | 2012             |
| validation   | any(defaul=2013) |
| test         | any              |
