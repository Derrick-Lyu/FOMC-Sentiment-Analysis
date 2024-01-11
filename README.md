## Project Description
 This project aims to predict Federal funds rate decision, which will hugely impact monetary policy and macroeconomic indexes in the US, from the Federal Open Market Committee's (FOMC) official documents.


## Dataset
Our datasets include both metadata and textual data.
- Metadata: macroeconomic variables from 1996 to 2023 organized by month, downloaded using FRED API,
- Textual data: Fed official documents recording texts to extract sentiments. Organized by meeting dates from 1999-2023 and downloaded using pypl and BeautifulSoup.
  
  Federal Minutes
  
  Federal Statement
  
  Federal Speech

## Preprocessing
The preprocessing included tokenization to break the text into words, removal of stopwords to eliminate less significant words, and lemmatization to standardize words to their base form. For sentiment analysis, the VADER tool was employed due to the absence of a suitable Loughran-McDonald dictionary, allowing for calculation of net sentiment scores for each FOMC statement. Additionally, the Federal Reserve's rate decisions were extracted and categorized as 1 for a rate increase, 0 for no change, and -1 (2) for a decrease, which were used as labels in our models.
Preprocessing and Exploratory Dataset Analysis can be found here:


## Models
- [Baseline ML](): In the classification task, metadata is used as independent variables to predict the Federal Reserve's rate decisions, which are the dependent variable. Baseline models including SVC, RFC, GBC, Perceptron, and AdaBoost are optimized using grid search and k-fold cross-validation to find the best hyperparameters. Due to an imbalanced dataset skewed towards 'hold' decisions, class weights are used to improve predictive performance by giving more importance to less frequent outcomes. 
- [LSTM](): we refined the LSTM model by incorporating Tfidf vectorization for input data. This enhancement, inspired by Takahashi (Takahashi, 2020), enables the model to retain more pertinent information, potentially improving its predictive accuracy. Additionally, we utilized Ray Tune Hyperband (Liaw et al., 2018), a technique explored in our coursework, for efficient hyperparameter optimization.
- [Baseline Fin-BERT](): We directly employed the Fin-BERT model (Huang et al., 2022), a variant of BERT specifically pre-trained on financial texts, as the BERT baseline. We applied this model to our aggregated dataset, which consists of FOMC minutes and statements analyzed at the sentence level. The model’s output was then used to calculate a weighted average sentiment score for each text paragraph. This score served as the basis for assigning sentiment labels to each paragraph, facilitating a more nuanced understanding of the underlying sentiment in financial communications. We later applied these sentiment scores to different classification models including Random Forest, SVC and KNN, to observe how these scores would contribute to rates decision classification. Hyper-parameter tuning is applied to these model to explore the best performance.
- [Fine-tuned Fin-BERT]() with passage summarization: The model incorporates advanced nlp NLP Processing Pipeline to understand and analyze text data at a granular level.

  - **NLTK Sentence Tokenizer (Paragraph Level)**: Utilizes NLTK's sentence tokenization to split text into individual sentences, facilitating sentence-level processing.

  - **BART**: Leverages the BART transformer model to perform sequence-to-sequence tasks, which performs summarization

  - **LED**: Employs the Longformer Encoder-Decoder model to efficiently process long sequences of text, making it suitable for comprehensive document analysis in this case (several 5000+sentence text)

  - **Fin-BERT**: Integrates a financial domain-specific BERT model, pre-trained on financial texts, and fin-tuned on out dataset for contextual and sentiment analysis in finance-related documents.
These components are sequentially applied to text data to extract and analyze information effectively, particularly suited for tasks in the federal announcement in financial domain.



## Conclusion
We ended up making improvements on previous work with an accuracy of around 94% by using a fine-tuned Fin-BERT model to input the textual paragraphs even if they are large.
| Model                  | Performance (Accuracy) |
|------------------------|------------------------|
| Baseline ML            | 0.877                  |
| LSTM                   | 0.729                  |
| Baseline Fin-BERT      | 0.864                  |
| Fine-tuned Fin-BERT    | 0.942                  |

## Interpretation

Form LSTM to Pre-Traiend Bert, we observe a huge increase in accuray. I think the reason is that LSTM is too simple that cannot understand the context, which is very important in the context of financial text, as the sentences are strongly connected. In contrast, by using the Bert model, which contain the self attention and multi head attention structure, naturally has better understanding on the context. Especially, the Fin-Bert is pretrained on financial dataset, which makes the model more favorable when we want to do the sentiment classification. Moreover, we fine-tuned the fin-bert model with our dataset, with selections of other financial statement dataset, with 100 epoch. Finally, we achieved 94.2% accuracy on compound dataset, and 98% accuracy on the target dataset, which is really impressive.

## Next Steps and Discussion:

We conducted an in-depth analysis of the Federal Open Market Committee data using advanced methodologies like machine learning, LSTM, FFNN, and an enhanced Fin-BERT model. I focused on matching sentiment variations in Federal Reserve statements to investment strategies, discovering crucial correlations.

A particular challenge was the small dataset size, inherent to FOMC data, which led to an imbalance, with a majority of data falling into the "Neutral" category. This affected model training and evaluation. I intend to incorporate more diverse sources to mitigate this and enhance the model's accuracy. Also, it is possible to break the long paragprahs into sentences. I have found that, by looking at the number of texts, we have only aroud 200 rows. However, if I break all the passages into the sentences, we have around 22,000 sentences, combining with federal statements from other english spoken country, we could finally end up with more than 100,000 sentences, which is enough to fine tune a more precise model that focus only at the financial statement made by government.

Reflecting on my work, I've successfully answered the initial research questions, yet there's a clear path ahead for deeper exploration and refinement.

## References
- [Kim's Paper](https://arxiv.org/abs/2304.10164)
- [Yuki Takahashi's blog](https://towardsdatascience.com/fedspeak-how-to-build-a-nlp-pipeline-to-predict-central-bank-policy-changes-a2f157ca0434)
- [Fin-BERT used](https://huggingface.co/yiyanghkust/finbert-tone)
- [Fine-tuned dataset used](https://huggingface.co/datasets/financial_phrasebank)
