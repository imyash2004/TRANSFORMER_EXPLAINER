// Text analysis utilities that work without real LLM models
export interface TokenData {
  id: number
  text: string
  color: string
  embedding: number[]
  pos_tag: string
  semantic_role: string
  explanation: string
  frequency: number
  sentiment: number
}

export interface AnalysisResult {
  tokens: TokenData[]
  category: string
  complexity: "beginner" | "intermediate" | "advanced"
  explanation: string
  predictedResponse: string
  confidence: number
}

// Simple tokenizer
export function tokenizeText(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 0)
}

// POS tagging simulation
export function getPOSTag(token: string, position: number, tokens: string[]): string {
  const questionWords = ["what", "how", "why", "when", "where", "who", "which"]
  const verbs = ["write", "create", "make", "explain", "tell", "show", "give", "find", "is", "are", "was", "were"]
  const nouns = ["poem", "story", "place", "time", "intelligence", "learning", "machine", "computer"]
  const adjectives = ["best", "good", "bad", "artificial", "simple", "complex", "beautiful"]
  const prepositions = ["on", "in", "at", "by", "for", "with", "about", "to", "from"]
  const determiners = ["the", "a", "an", "this", "that", "these", "those"]

  if (questionWords.includes(token)) return "PRON"
  if (verbs.includes(token)) return "VERB"
  if (nouns.includes(token)) return "NOUN"
  if (adjectives.includes(token)) return "ADJ"
  if (prepositions.includes(token)) return "PREP"
  if (determiners.includes(token)) return "DET"
  if (token.charAt(0) === token.charAt(0).toUpperCase()) return "PROPN"
  return "NOUN" // default
}

// Semantic role assignment
export function getSemanticRole(token: string, posTag: string, position: number): string {
  const roleMap: { [key: string]: string } = {
    PRON: "QUESTION",
    VERB: "ACTION",
    NOUN: "ENTITY",
    ADJ: "ATTRIBUTE",
    PREP: "RELATION",
    DET: "DETERMINER",
    PROPN: "PLACE",
  }
  return roleMap[posTag] || "OTHER"
}

// Generate realistic embeddings based on semantic similarity
export function generateEmbedding(token: string, semanticRole: string): number[] {
  const baseVectors: { [key: string]: number[] } = {
    QUESTION: [0.3, 0.7, 0.4, 0.2, 0.8],
    ACTION: [0.6, 0.3, 0.7, 0.9, 0.2],
    ENTITY: [0.4, 0.8, 0.2, 0.5, 0.6],
    ATTRIBUTE: [0.7, 0.2, 0.8, 0.3, 0.5],
    RELATION: [0.2, 0.5, 0.3, 0.7, 0.4],
    DETERMINER: [0.1, 0.3, 0.2, 0.4, 0.1],
    PLACE: [0.8, 0.6, 0.9, 0.4, 0.7],
  }

  const base = baseVectors[semanticRole] || [0.5, 0.5, 0.5, 0.5, 0.5]

  // Add some noise based on token characteristics
  const tokenHash = token.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0)
  const noise = Array.from({ length: 5 }, (_, i) => Math.sin(tokenHash + i) * 0.2)

  return base.map((val, i) => Math.max(0, Math.min(1, val + noise[i])))
}

// Analyze any input text
export function analyzeText(text: string): AnalysisResult {
  const tokens = tokenizeText(text)
  const colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#FF9FF3", "#F093FB", "#4FACFE"]

  const tokenData: TokenData[] = tokens.map((token, index) => {
    const posTag = getPOSTag(token, index, tokens)
    const semanticRole = getSemanticRole(token, posTag, index)
    const embedding = generateEmbedding(token, semanticRole)

    return {
      id: index,
      text: token,
      color: colors[index % colors.length],
      embedding,
      pos_tag: posTag,
      semantic_role: semanticRole,
      explanation: generateTokenExplanation(token, posTag, semanticRole),
      frequency: Math.random() * 0.1 + 0.01, // Simulated frequency
      sentiment: Math.random() * 2 - 1, // -1 to 1
    }
  })

  const category = categorizeText(text)
  const complexity = assessComplexity(text, tokenData)
  const explanation = generateExplanation(text, category, complexity)
  const predictedResponse = generatePredictedResponse(text, category)
  const confidence = calculateConfidence(text, tokenData)

  return {
    tokens: tokenData,
    category,
    complexity,
    explanation,
    predictedResponse,
    confidence,
  }
}

function generateTokenExplanation(token: string, posTag: string, semanticRole: string): string {
  const explanations: { [key: string]: string } = {
    PRON: `"${token}" is a question word that signals the user is asking for information`,
    VERB: `"${token}" is an action word that tells the model what task to perform`,
    NOUN: `"${token}" is a thing or concept that provides context for the request`,
    ADJ: `"${token}" is a descriptive word that adds specific qualities or attributes`,
    PREP: `"${token}" is a connecting word that shows relationships between other words`,
    DET: `"${token}" is a determiner that specifies quantity or definiteness`,
    PROPN: `"${token}" is a proper noun, likely a specific name or place`,
  }
  return explanations[posTag] || `"${token}" contributes to the overall meaning of the text`
}

function categorizeText(text: string): string {
  const lowerText = text.toLowerCase()

  if (lowerText.includes("write") || lowerText.includes("poem") || lowerText.includes("story")) {
    return "Creative Writing"
  } else if (lowerText.includes("what") || lowerText.includes("how") || lowerText.includes("why")) {
    return "Question & Answer"
  } else if (lowerText.includes("explain") || lowerText.includes("define") || lowerText.includes("meaning")) {
    return "Educational"
  } else if (lowerText.includes("best") || lowerText.includes("recommend") || lowerText.includes("place")) {
    return "Recommendation"
  } else if (lowerText.includes("once upon") || lowerText.includes("story") || lowerText.includes("tale")) {
    return "Storytelling"
  }

  return "General Query"
}

function assessComplexity(text: string, tokens: TokenData[]): "beginner" | "intermediate" | "advanced" {
  const avgTokenLength = tokens.reduce((sum, token) => sum + token.text.length, 0) / tokens.length
  const uniqueRoles = new Set(tokens.map((t) => t.semantic_role)).size

  if (tokens.length <= 4 && avgTokenLength <= 5 && uniqueRoles <= 3) {
    return "beginner"
  } else if (tokens.length <= 8 && avgTokenLength <= 7 && uniqueRoles <= 5) {
    return "intermediate"
  }
  return "advanced"
}

function generateExplanation(text: string, category: string, complexity: string): string {
  const explanations = {
    "Creative Writing":
      "This prompt asks the AI to generate creative content. The model will use its training on literature and creative texts to produce original content.",
    "Question & Answer":
      "This is an information-seeking query. The model will search its knowledge base to provide factual, helpful answers.",
    Educational:
      "This prompt requests an explanation or educational content. The model will break down complex topics into understandable parts.",
    Recommendation:
      "This asks for suggestions or recommendations. The model will use its knowledge to provide helpful suggestions based on the context.",
    Storytelling:
      "This prompt initiates a narrative. The model will use storytelling patterns and structures learned from its training data.",
    "General Query":
      "This is a general request that the model will process using its broad knowledge and language understanding capabilities.",
  }

  return explanations[category] || "The model will process this text using its language understanding capabilities."
}

function generatePredictedResponse(text: string, category: string): string {
  const responses = {
    "Creative Writing":
      "The model would generate creative content like poetry, stories, or artistic descriptions based on the prompt.",
    "Question & Answer": "The model would provide a factual, informative answer drawing from its training knowledge.",
    Educational: "The model would break down the concept into simple, understandable explanations with examples.",
    Recommendation: "The model would suggest relevant options based on the context and criteria mentioned.",
    Storytelling: "The model would continue or create a narrative with characters, plot, and descriptive elements.",
    "General Query": "The model would provide a helpful response tailored to the specific request.",
  }

  return responses[category] || "The model would generate an appropriate response based on the input context."
}

function calculateConfidence(text: string, tokens: TokenData[]): number {
  // Simple confidence calculation based on token clarity and common patterns
  const commonTokens = tokens.filter((t) => ["VERB", "NOUN", "ADJ"].includes(t.pos_tag)).length
  const totalTokens = tokens.length

  return Math.min(0.95, 0.6 + (commonTokens / totalTokens) * 0.3 + Math.random() * 0.1)
}

export const EXAMPLES = [
  {
    id: 1,
    text: "write a poem on winter",
    category: "Creative Writing",
    icon: "❄️",
    gradient: "from-blue-400 to-blue-600",
    description: "Generate creative poetry about winter themes",
    difficulty: "Beginner",
    explanation:
      "This is a creative writing task. The AI needs to understand 'write' as an instruction, 'poem' as the format, and 'winter' as the theme. It will use its knowledge of poetry structure and winter imagery to create original content.",
  },
  {
    id: 2,
    text: "what is the capital of France",
    category: "Factual Question",
    icon: "🗼",
    gradient: "from-green-400 to-green-600",
    description: "Simple factual question requiring specific knowledge",
    difficulty: "Beginner",
    explanation:
      "This is a straightforward factual question. The AI recognizes 'what is' as a question pattern, 'capital' as a geographic concept, and 'France' as a country. It retrieves the stored fact that Paris is France's capital.",
  },
  {
    id: 3,
    text: "explain machine learning to a 5 year old",
    category: "Educational",
    icon: "🧠",
    gradient: "from-purple-400 to-purple-600",
    description: "Complex concept simplified for young audience",
    difficulty: "Intermediate",
    explanation:
      "This requires the AI to understand multiple concepts: 'explain' (instruction), 'machine learning' (complex topic), and 'to a 5 year old' (audience specification). It must simplify technical concepts using analogies and simple language.",
  },
  {
    id: 4,
    text: "translate hello world to Spanish",
    category: "Translation",
    icon: "🌍",
    gradient: "from-orange-400 to-orange-600",
    description: "Language translation task",
    difficulty: "Beginner",
    explanation:
      "Translation task where the AI identifies 'translate' as the action, 'hello world' as the source text, and 'Spanish' as the target language. It uses its multilingual training to provide the equivalent phrase.",
  },
  {
    id: 5,
    text: "compare and contrast cats and dogs as pets",
    category: "Analysis",
    icon: "🐱",
    gradient: "from-red-400 to-red-600",
    description: "Analytical comparison requiring structured thinking",
    difficulty: "Advanced",
    explanation:
      "This complex task requires the AI to understand 'compare and contrast' as an analytical framework, identify 'cats and dogs' as subjects, and 'as pets' as the context. It must organize information into similarities and differences.",
  },
  {
    id: 6,
    text: "once upon a time in a magical forest",
    category: "Storytelling",
    icon: "🌲",
    gradient: "from-emerald-400 to-emerald-600",
    description: "Story beginning that triggers narrative generation",
    difficulty: "Intermediate",
    explanation:
      "This storytelling prompt uses the classic 'once upon a time' opening to signal narrative mode. The AI recognizes this pattern and prepares to generate a story with characters, setting, and plot development.",
  },
]
