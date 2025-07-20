# Fashion Dataset EDA Report

## Data Loading and Cleaning

```
=== Loading Raw Data ===
Raw dataset count: 44446
Count after dropping NA: 44424
Count after filtering valid IDs: 44424
Final count after all cleaning: 44410
```

**Insight**: The dataset starts with 44446 records, with 36 rows dropped due to missing values (22 rows) and empty strings (14 rows), resulting in a final count of 44410. This minor data loss suggests robust data quality but highlights the need to handle missing or invalid entries carefully.

## Dataset Info

```
<class 'pandas.core.frame.DataFrame'>
Index: 44410 entries
Data columns (total 5 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   id           44410 non-null  int64 
 1   gender       44410 non-null  object
 2   baseColour   44410 non-null  object
 3   articleType  44410 non-null  object
 4   season       44410 non-null  object
dtypes: int64(1), object(4)
memory usage: 2.0+ MB
```

## Numerical Columns Summary

```
                 id
count   44410.000000
mean    29781.378068
std     17166.345863
min         1.000000
25%     14881.250000
50%     29810.500000
75%     44713.750000
max     60000.000000
```

## Categorical Columns Summary

```
         gender articleType baseColour  season
count    44410      44410      44410   44410
unique       5        143         46       4
top        Men    Tshirts      Black  Summer
freq     22146       7069       9728   21476
```

**Insight**: The dataset has 5 genders, 143 article types, 46 base colours, and 4 seasons. The high number of article types (143) and colours (46) indicates complexity, with significant imbalances (e.g., Tshirts: 7069, Black: 9728, Summer: 21476 dominate their respective categories).

## Summary Statistics

```
Total Records: 44410
Unique Genders: 5 (Men, Women, Unisex, Boys, Girls)
Unique Seasons: 4 (Fall, Summer, Winter, Spring)
Unique Article Types: 143
Unique Base Colours: 46
```

## Interesting Fact

```
Most popular color for Men: Black (4488 items)
Most popular color for Women: Black (3463 items)
Most popular color for Unisex: Black (657 items)
Most popular color for Boys: Blue (627 items)
Most popular color for Girls: Pink (352 items)
```

**Insight**: Black dominates as the most popular color across Men, Women, and Unisex, while Boys prefer Blue and Girls prefer Pink, suggesting gender-specific marketing opportunities.

## Data Imbalance Analysis

- **Gender Imbalance**:
  - Men: 22146 (49.8%)
  - Women: 16553 (37.3%)
  - Unisex: 2418 (5.4%)
  - Boys: 2558 (5.8%)
  - Girls: 1735 (3.9%)
  - **Insight**: Men and Women dominate (\~87% of data), while Unisex, Boys, and Girls are underrepresented, potentially leading to biased model predictions for minority classes.
- **Color Imbalance**:
  - Top 10 colors (Black, Blue, White, Grey, Red, Green, Navy Blue, Brown, Pink, Purple) cover \~82% of the data (36344 items).
  - Remaining 36 colors cover only \~18% (8066 items).
  - **Insight**: The skewed distribution toward top colors suggests grouping into color families (e.g., Neutral: Black/White/Grey, Blue: Blue/Navy Blue) to reduce complexity and handle real-world color variations.
- **Season Imbalance**:
  - Summer: 21476 (48.3%)
  - Fall: 14479 (32.6%)
  - Winter: 6808 (15.3%)
  - Spring: 1647 (3.7%)
  - **Insight**: Summer and Fall dominate (\~81%), while Spring is severely underrepresented, which may affect seasonal trend predictions.
- **Article Type Imbalance**:
  - Top type (Tshirts: 7069) vs. rare types (some with &lt;20 items).
  - **Insight**: The 143 article types create high model complexity, with dominant types like Tshirts and Shirts skewing predictions.

**Mitigation Strategies**:

- **Class Weights**: Apply inverse frequency weights in the loss function to prioritize minority classes (e.g., Girls, Spring).
- **Oversampling**: Oversampling for minority classes like Girls and Spring.
- **Color Families**: Group 46 colors into color-families.
- **Article Type Classification**: Use masterCategory Or subCategory in place of Article Type to reduce model complexity.
- **Augmentation**: Apply image augmentation for minority classes to increase diversity.

## Visualizations

### Distribution of Products by Gender

**Description**: Bar chart showing the distribution of products by gender, with Men and Women as the largest categories.

![Gender Distribution chart](https://github.com/AnkitSharma1405/Fashion_Dataset_Assignment/blob/9848e46b5b93cf57bc0170e33c0c83f1e0502f54/EDA/Distribution_charts/gender_distribution.png)

**Data Table**:

| Gender | Count | Percentage |  |
| --- | --- | --- | --- |
| Men | 22146 | 49.8% |  |
| Women | 16553 | 37.3% |  |
| Unisex | 2418 | 5.4% |  |
| Boys | 2558 | 5.8% |  |
| Girls | 1735 | 3.9% |  |

**Insight**: Men and Women dominate (\~87%), indicating a significant imbalance. Class weights or oversampling are needed to improve predictions for Unisex, Boys, and Girls.

### Top 10 Base Colours

**Description**: Bar chart showing the top 10 base colours, covering \~82% of the dataset.

![Color Distribution chart](https://github.com/AnkitSharma1405/Fashion_Dataset_Assignment/blob/9848e46b5b93cf57bc0170e33c0c83f1e0502f54/EDA/Distribution_charts/color_distribution.png)

**Data Table**:

| Color | Count | Percentage |  |
| --- | --- | --- | --- |
| Black | 9728 | 21.9% |  |
| Blue | 6956 | 15.7% |  |
| White | 5537 | 12.5% |  |
| Grey | 4058 | 9.1% |  |
| Red | 2838 | 6.4% |  |
| Green | 2460 | 5.5% |  |
| Navy Blue | 2418 | 5.4% |  |
| Brown | 2333 | 5.3% |  |
| Pink | 1746 | 3.9% |  |
| Purple | 1470 | 3.3% |  |

**Insight**: The top 10 colours cover \~82% of the data, suggesting that grouping the 46 colors into families (e.g., Neutral, Blue, Red) can reduce complexity and improve model robustness.

### Proportion of Products by Season

**Description**: Pie chart showing the proportion of products by season.

![session Distribution chart](https://github.com/AnkitSharma1405/Fashion_Dataset_Assignment/blob/9848e46b5b93cf57bc0170e33c0c83f1e0502f54/EDA/Distribution_charts/session_distribution.png)

**Data Table**:

| Season | Count | Percentage |  |
| --- | --- | --- | --- |
| Summer | 21476 | 48.3% |  |
| Fall | 14479 | 32.6% |  |
| Winter | 6808 | 15.3% |  |
| Spring | 1647 | 3.7% |  |

**Insight**: Summer and Fall dominate (\~81%), while Spring’s low representation (3.7%) necessitates oversampling or class weights to avoid biased seasonal predictions.

### Article Type Distribution by Season

**Description**: Stacked bar chart showing the distribution of top 5 article types (Tshirts, Shirts, Tops, Kurtas, Watches) across seasons.

![Type Distribution chart](https://github.com/AnkitSharma1405/Fashion_Dataset_Assignment/blob/9848e46b5b93cf57bc0170e33c0c83f1e0502f54/EDA/Distribution_charts/type_distribution.png)

**Data Table**:

| Season | Tshirts | Shirts | Tops | Kurtas | Watches |
| --- | --- | --- | --- | --- | --- |
| Fall | 2309 | 1757 | 1011 | 644 | 426 |
| Spring | 578 | 192 | 205 | 115 | 89 |
| Summer | 3164 | 2104 | 1644 | 1098 | 774 |
| Winter | 1018 | 614 | 566 | 329 | 2157 |

**Text Representation (scaled, 1 '=' \~500 items)**:

**Insight**: Tshirts and Shirts dominate across seasons, but the high number of article types (143) increases model complexity, suggesting the use of masterCategory/subCategory for hierarchical classification.

## Summary Table

| Feature | Unique Values | Count |
| --- | --- | --- |
| Gender | Men, Women, Unisex, Boys, Girls | 5 |
| Season | Fall, Summer, Winter, Spring | 4 |
| Article Type | Tshirts, Shirts, Tops, Kurtas, Watches | 143 |
| Base Colour | Black, Blue, White, Grey, Red, Green, Navy Blue, Brown, Pink, Purple | 46 |

## Conclusion


The EDA on the cleaned dataset with 44410 records reveals a skewed gender distribution, with Men (22146, 49.8%) and Women (16553, 37.3%) dominating (~87%), while Unisex (5.4%), Boys (5.8%), and Girls (3.9%) are underrepresented. Summer (21476, 48.3%) and Fall (14479, 32.6%) are key seasons, with Spring (1647, 3.7%) severely underrepresented. T-shirts (7069) and Shirts are top article types among 143, and Black (9728), Blue, and White dominate among 46 colors, with top 10 colors covering ~82% of the data. The high number of article types and colors, combined with imbalances, suggests grouping colors into families (e.g., Neutral, Blue) and using masterCategory/subCategory for hierarchical classification. Class weights, oversampling, and augmentation are recommended to address imbalances and improve model performance.
