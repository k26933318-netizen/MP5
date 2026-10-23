function cosineSimilarity(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== 512 || right.length !== 512 ||
      left.some(value => !Number.isFinite(value)) || right.some(value => !Number.isFinite(value))) {
    const error = new Error('Face embedding is invalid');
    error.code = 'INVALID_FACE_EMBEDDING';
    error.statusCode = 502;
    throw error;
  }
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < 512; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] * left[index];
    rightNorm += right[index] * right[index];
  }
  if (leftNorm === 0 || rightNorm === 0) {
    const error = new Error('Face embedding is invalid');
    error.code = 'INVALID_FACE_EMBEDDING';
    error.statusCode = 502;
    throw error;
  }
  return dot / Math.sqrt(leftNorm * rightNorm);
}

function faceSimilarityThreshold() {
  const value = Number(process.env.FACE_SIMILARITY_THRESHOLD || 0.5);
  if (!Number.isFinite(value) || value <= 0 || value > 1) {
    throw new Error('FACE_SIMILARITY_THRESHOLD must be between 0 and 1');
  }
  return value;
}

function findMatchingEmbedding(candidate, embeddings, threshold) {
  let match = null;
  for (let index = 0; index < embeddings.length; index += 1) {
    const score = cosineSimilarity(candidate, embeddings[index]);
    if (score >= threshold && (!match || score > match.score)) {
      match = { index, score };
    }
  }
  return match;
}

module.exports = { cosineSimilarity, faceSimilarityThreshold, findMatchingEmbedding };
