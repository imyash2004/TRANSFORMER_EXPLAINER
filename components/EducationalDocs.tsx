"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { 
  BookOpen, 
  Brain, 
  Calculator, 
  Eye, 
  Network, 
  Zap,
  Target,
  Lightbulb,
  Info,
  TrendingUp,
  Globe,
  Code,
  AlertTriangle
} from "lucide-react"

interface EducationalContent {
  title: string
  overview: string
  keyPoints: string[]
  examples: string[]
  mathematics?: string
  realWorld: string[]
  commonMisconceptions: string[]
}

const educationalSections: Record<string, EducationalContent> = {
  tokens: {
    title: "Understanding Tokens: The Building Blocks of Language Processing",
    overview: "Tokens are NOT words. They are the fundamental atomic units that language models use to process text, representing words, parts of words, punctuation, or special symbols that have no direct correspondence to human language concepts.",
    keyPoints: [
      "Tokens are created through Byte Pair Encoding (BPE) algorithm",
      "Common words get their own tokens, rare words are split into subwords",
      "Modern LLMs use 50K-100K token vocabularies",
      "Tokenization affects model performance and handling of rare words",
      "Special tokens like <pad>, <unk>, <s>, </s> control model behavior"
    ],
    examples: [
      '"the" → Token ID: 464 (whole word)',
      '"unhappiness" → ["un", "happy", "ness"] (subword split)',
      '"ChatGPT-4o" → ["Chat", "GPT", "-", "4", "o"] (technical term handling)',
      '"こんにちは" → ["こん", "にちは"] (multilingual support)'
    ],
    mathematics: "token_ids = tokenizer.encode(text) # BPE algorithm merges most frequent character pairs",
    realWorld: [
      "GPT-4 uses ~100,000 tokens for optimal efficiency",
      "Vocabulary size balances memory usage vs expressiveness",
      "Subword tokenization enables handling of new/rare words",
      "Cross-lingual models share tokens across languages"
    ],
    commonMisconceptions: [
      "❌ Tokens are not words - they can be smaller or larger than words",
      "❌ Tokenization is not arbitrary - it's learned from training data",
      "❌ All languages don't tokenize the same way",
      "❌ Longer text doesn't always mean more tokens"
    ]
  },
  
  embeddings: {
    title: "From Words to Numbers: The Mathematical World of Embeddings",
    overview: "Embeddings transform discrete tokens into continuous mathematical vectors that capture semantic meaning, enabling computers to perform arithmetic operations on language concepts.",
    keyPoints: [
      "Each token maps to a high-dimensional vector (768-12,288 dimensions)",
      "Similar tokens have similar vectors in embedding space",
      "Embeddings enable mathematical operations on meaning",
      "Vector arithmetic captures semantic relationships",
      "Embeddings are learned, not programmed"
    ],
    examples: [
      'king - man + woman ≈ queen (gender relationships)',
      'Paris - France + Germany ≈ Berlin (country-capital)',
      'walking - walk + swim ≈ swimming (verb tense)',
      'cat ↔ dog similarity = 0.73 (high semantic similarity)'
    ],
    mathematics: "embedding = E[token_id] # where E is learned matrix [vocab_size × embedding_dim]",
    realWorld: [
      "GPT-4 embeddings: 100K × 12,288 = 1.2B parameters just for embeddings",
      "Embeddings capture cultural and linguistic biases from training data",
      "Transfer learning leverages pre-trained embeddings",
      "Embedding spaces can be visualized using dimensionality reduction"
    ],
    commonMisconceptions: [
      "❌ Embeddings are not manually designed - they emerge from training",
      "❌ Higher dimensions don't always mean better performance",
      "❌ Embeddings don't directly encode human-interpretable features",
      "❌ Similar embeddings don't guarantee similar model behavior"
    ]
  },

  attention: {
    title: "Attention Mechanism: How Models Focus and Relate Information",
    overview: "The attention mechanism allows models to selectively focus on relevant parts of the input when processing each token, using Query-Key-Value operations to compute weighted combinations of information.",
    keyPoints: [
      "Attention uses Query (what I need), Key (what I have), Value (actual content)",
      "Self-attention allows tokens to attend to other tokens in the sequence",
      "Multi-head attention captures different types of relationships",
      "Attention weights are learned, not programmed",
      "Parallel processing enables efficient computation"
    ],
    examples: [
      'In "The cat sat", "cat" attends strongly to "sat" (subject-verb)',
      'Multi-head attention: syntax, semantics, position, dependencies',
      'Long-range dependencies: connecting distant but related words',
      'Attention weights visualize model focus patterns'
    ],
    mathematics: "Attention(Q,K,V) = softmax(QK^T/√d_k)V",
    realWorld: [
      "GPT-4 has 128 attention heads across 96 layers",
      "Attention patterns reveal model reasoning strategies",
      "Different heads specialize in different linguistic phenomena",
      "Attention enables handling of variable-length sequences"
    ],
    commonMisconceptions: [
      "❌ Attention is not the same as human attention",
      "❌ High attention doesn't always mean importance",
      "❌ Attention patterns don't always correspond to linguistic intuition",
      "❌ Self-attention is not circular reasoning"
    ]
  },

  process: {
    title: "Complete LLM Processing Pipeline: From Input to Output",
    overview: "LLMs process text through a series of mathematical transformations: tokenization → embeddings → positional encoding → multiple transformer layers → final prediction.",
    keyPoints: [
      "Processing is parallel, not sequential like human reading",
      "Each layer refines token representations",
      "Residual connections enable deep networks",
      "Layer normalization stabilizes training",
      "Final prediction considers entire vocabulary simultaneously"
    ],
    examples: [
      'Input: "The weather is" → Tokens: [464, 6193, 318]',
      'Each token becomes 12,288-dimensional vector',
      '96 transformer layers progressively refine understanding',
      'Final layer outputs probabilities for all 100K possible next tokens'
    ],
    mathematics: "layer_out = LayerNorm(x + MultiHeadAttention(x)) + LayerNorm(x + FFN(x))",
    realWorld: [
      "GPT-4 has 1.76 trillion parameters across 96 layers",
      "Processing requires massive parallel computation (GPUs/TPUs)",
      "Inference costs scale with sequence length and model size",
      "Model size correlates with capability (scaling laws)"
    ],
    commonMisconceptions: [
      "❌ LLMs don't understand like humans do",
      "❌ Bigger models aren't always better for all tasks",
      "❌ LLMs don't have consciousness or sentience",
      "❌ Processing isn't deterministic - sampling introduces randomness"
    ]
  }
}

interface EducationalDocsProps {
  activeSection?: string
}

export function EducationalDocs({ activeSection = "tokens" }: EducationalDocsProps) {
  const [selectedSection, setSelectedSection] = useState(activeSection)
  const [showMath, setShowMath] = useState(false)

  const sections = Object.keys(educationalSections)
  const currentContent = educationalSections[selectedSection]

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <BookOpen className="h-6 w-6" />
              <span>LLM Educational Documentation</span>
            </CardTitle>
            <div className="flex items-center space-x-2">
              <Button
                variant={showMath ? "default" : "outline"}
                size="sm"
                onClick={() => setShowMath(!showMath)}
              >
                <Calculator className="h-4 w-4 mr-2" />
                Math Mode
              </Button>
              <Badge variant="outline">Interactive Guide</Badge>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Section Navigation */}
      <Card>
        <CardContent className="p-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { key: "tokens", icon: Zap, title: "Tokenization", desc: "Text to tokens" },
              { key: "embeddings", icon: Brain, title: "Embeddings", desc: "Vectors & meaning" },
              { key: "attention", icon: Network, title: "Attention", desc: "QKV mechanism" },
              { key: "process", icon: Target, title: "Full Pipeline", desc: "End-to-end flow" }
            ].map(({ key, icon: Icon, title, desc }) => (
              <button
                key={key}
                onClick={() => setSelectedSection(key)}
                className={`p-4 rounded-lg text-left transition-all ${
                  selectedSection === key
                    ? "bg-blue-100 dark:bg-blue-900 ring-2 ring-blue-500"
                    : "hover:bg-gray-100 dark:hover:bg-gray-700"
                }`}
              >
                <div className="flex items-center space-x-3">
                  <Icon className="h-5 w-5 text-blue-600" />
                  <div>
                    <div className="font-medium text-sm">{title}</div>
                    <div className="text-xs text-gray-500">{desc}</div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Main Content */}
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">{currentContent.title}</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="overview" className="w-full">
            <TabsList className="grid w-full grid-cols-6">
              <TabsTrigger value="overview" className="flex items-center space-x-1">
                <Eye className="h-3 w-3" />
                <span className="hidden sm:inline">Overview</span>
              </TabsTrigger>
              <TabsTrigger value="keypoints" className="flex items-center space-x-1">
                <Lightbulb className="h-3 w-3" />
                <span className="hidden sm:inline">Key Points</span>
              </TabsTrigger>
              <TabsTrigger value="examples" className="flex items-center space-x-1">
                <Code className="h-3 w-3" />
                <span className="hidden sm:inline">Examples</span>
              </TabsTrigger>
              <TabsTrigger value="mathematics" className="flex items-center space-x-1">
                <Calculator className="h-3 w-3" />
                <span className="hidden sm:inline">Math</span>
              </TabsTrigger>
              <TabsTrigger value="realworld" className="flex items-center space-x-1">
                <Globe className="h-3 w-3" />
                <span className="hidden sm:inline">Real World</span>
              </TabsTrigger>
              <TabsTrigger value="misconceptions" className="flex items-center space-x-1">
                <AlertTriangle className="h-3 w-3" />
                <span className="hidden sm:inline">Myths</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-4 mt-6">
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <h3 className="font-medium text-blue-800 dark:text-blue-200 mb-2">Core Concept</h3>
                <p className="text-sm text-blue-700 dark:text-blue-300">{currentContent.overview}</p>
              </div>
              
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <h3 className="font-medium mb-2">Why This Matters</h3>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  {selectedSection === "tokens" && "Understanding tokenization is crucial because it affects every aspect of model behavior - from handling rare words to computational efficiency. Poor tokenization can cause models to struggle with certain languages or technical terms."}
                  {selectedSection === "embeddings" && "Embeddings are the foundation that makes all language model capabilities possible. They transform the discrete, symbolic nature of language into continuous mathematical space where similarity, analogy, and semantic relationships can be computed."}
                  {selectedSection === "attention" && "The attention mechanism is what allows transformers to understand context and relationships in language. It's the key innovation that enabled the breakthrough in natural language understanding we see in modern LLMs."}
                  {selectedSection === "process" && "Understanding the complete processing pipeline helps demystify how LLMs work and reveals both their capabilities and limitations. This knowledge is essential for effective use and responsible deployment of these systems."}
                </p>
              </div>
            </TabsContent>

            <TabsContent value="keypoints" className="space-y-3 mt-6">
              {currentContent.keyPoints.map((point, index) => (
                <div key={index} className="flex items-start space-x-3 p-3 bg-white dark:bg-gray-800 rounded-lg border">
                  <div className="w-6 h-6 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center flex-shrink-0 mt-0.5">
                    {index + 1}
                  </div>
                  <span className="text-sm">{point}</span>
                </div>
              ))}
            </TabsContent>

            <TabsContent value="examples" className="space-y-4 mt-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {currentContent.examples.map((example, index) => (
                  <div key={index} className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <div className="font-mono text-sm text-green-800 dark:text-green-200">{example}</div>
                  </div>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="mathematics" className="space-y-4 mt-6">
              {currentContent.mathematics && (
                <div className="space-y-4">
                  <div className="p-4 bg-gray-100 dark:bg-gray-700 rounded-lg">
                    <h3 className="font-medium mb-2">Mathematical Formula</h3>
                    <div className="font-mono text-sm bg-white dark:bg-gray-800 p-3 rounded border">
                      {currentContent.mathematics}
                    </div>
                  </div>

                  <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                    <h3 className="font-medium text-purple-800 dark:text-purple-200 mb-2">Detailed Explanation</h3>
                    <div className="text-sm text-purple-700 dark:text-purple-300 space-y-2">
                      {selectedSection === "tokens" && (
                        <div>
                          <p><strong>BPE Algorithm Steps:</strong></p>
                          <ol className="list-decimal list-inside space-y-1 ml-4">
                            <li>Start with character-level tokenization</li>
                            <li>Count frequency of adjacent character pairs</li>
                            <li>Merge most frequent pair into new token</li>
                            <li>Repeat until reaching desired vocabulary size</li>
                          </ol>
                        </div>
                      )}
                      {selectedSection === "embeddings" && (
                        <div>
                          <p><strong>Embedding Matrix Lookup:</strong></p>
                          <p>E is a learned matrix where each row corresponds to a token's vector representation. The lookup operation E[token_id] retrieves the dense vector for that token.</p>
                        </div>
                      )}
                      {selectedSection === "attention" && (
                        <div>
                          <p><strong>QKV Attention Steps:</strong></p>
                          <ol className="list-decimal list-inside space-y-1 ml-4">
                            <li>Compute Q·K^T (query-key dot products)</li>
                            <li>Scale by √d_k to prevent vanishing gradients</li>
                            <li>Apply softmax to get attention weights</li>
                            <li>Multiply weights by Values to get output</li>
                          </ol>
                        </div>
                      )}
                      {selectedSection === "process" && (
                        <div>
                          <p><strong>Transformer Layer Components:</strong></p>
                          <p>Each layer applies multi-head attention and feed-forward networks with residual connections and layer normalization for stable training.</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="realworld" className="space-y-3 mt-6">
              {currentContent.realWorld.map((fact, index) => (
                <div key={index} className="flex items-start space-x-3 p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                  <TrendingUp className="h-4 w-4 text-orange-600 flex-shrink-0 mt-0.5" />
                  <span className="text-sm text-orange-800 dark:text-orange-200">{fact}</span>
                </div>
              ))}
            </TabsContent>

            <TabsContent value="misconceptions" className="space-y-3 mt-6">
              {currentContent.commonMisconceptions.map((misconception, index) => (
                <div key={index} className="flex items-start space-x-3 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                  <AlertTriangle className="h-4 w-4 text-red-600 flex-shrink-0 mt-0.5" />
                  <span className="text-sm text-red-800 dark:text-red-200">{misconception}</span>
                </div>
              ))}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Quick Reference */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Info className="h-5 w-5" />
            <span>Quick Reference</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
              <div className="font-medium text-sm text-blue-800 dark:text-blue-200">Vocab Size</div>
              <div className="text-xs text-blue-600 dark:text-blue-300">GPT-4: ~100K tokens</div>
            </div>
            <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
              <div className="font-medium text-sm text-green-800 dark:text-green-200">Embedding Dim</div>
              <div className="text-xs text-green-600 dark:text-green-300">GPT-4: 12,288 dimensions</div>
            </div>
            <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
              <div className="font-medium text-sm text-purple-800 dark:text-purple-200">Attention Heads</div>
              <div className="text-xs text-purple-600 dark:text-purple-300">GPT-4: 128 heads</div>
            </div>
            <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
              <div className="font-medium text-sm text-orange-800 dark:text-orange-200">Layers</div>
              <div className="text-xs text-orange-600 dark:text-orange-300">GPT-4: 96 transformer layers</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
