import { Document } from "@langchain/core/documents";
import {
  VectorStore,
  VectorStoreRetriever,
  VectorStoreRetrieverInput,
} from "@langchain/core/vectorstores";

export type ScoreThresholdRetrieverInput<V extends VectorStore> = Omit<
  VectorStoreRetrieverInput<V>,
  "k"
> & {
  maxK?: number;
  kIncrement?: number;
  minSimilarityScore?: number;
  maxDistanceScore?: number;
};

export class ScoreThresholdRetriever<
  V extends VectorStore
> extends VectorStoreRetriever<V> {
  minSimilarityScore: number;

  maxDistanceScore: number;

  kIncrement = 10;

  maxK = 100;

  constructor(input: ScoreThresholdRetrieverInput<V>) {
    super(input);
    this.maxK = input.maxK ?? this.maxK;
    this.minSimilarityScore =
      input.minSimilarityScore ?? this.minSimilarityScore;
    this.maxDistanceScore = input.maxDistanceScore ?? this.maxDistanceScore;

    if (!this.minSimilarityScore && !this.maxDistanceScore) {
      throw new Error(
        "At least minSimilarityScore or maxDistanceScore must be provided"
      );
    }

    this.kIncrement = input.kIncrement ?? this.kIncrement;
  }

  async getRelevantDocuments(query: string): Promise<Document[]> {
    let currentK = 0;
    let filteredResults: [Document, number][] = [];
    do {
      currentK += this.kIncrement;
      const results = await this.vectorStore.similaritySearchWithScore(
        query,
        currentK,
        this.filter
      );
      filteredResults = results.filter(([, score]) => {
        if (this.minSimilarityScore !== undefined) {
          return score >= this.minSimilarityScore;
        } else if (this.maxDistanceScore !== undefined) {
          return score <= this.maxDistanceScore;
        }
        return false;
      });
    } while (filteredResults.length >= currentK && currentK < this.maxK);
    return filteredResults.map((documents) => documents[0]).slice(0, this.maxK);
  }

  static fromVectorStore<V extends VectorStore>(
    vectorStore: V,
    options: Omit<ScoreThresholdRetrieverInput<V>, "vectorStore">
  ) {
    return new this<V>({ ...options, vectorStore });
  }
  static test<V extends VectorStore>(vectorStore: V) {
    return new this<V>({
      vectorStore: vectorStore,
      minSimilarityScore: 0.5,
    });
  }
}
