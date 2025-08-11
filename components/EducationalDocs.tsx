"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import MultiHeadAttentionVisualizer from "./MultiHeadAttentionVisualizer"
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
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Layers,
  Settings,
  HelpCircle,
  FileText,
  Microscope,
  Cpu,
  Scale
} from "lucide-react"

interface FAQ {
  question: string
  answer: string
  category: "beginner" | "intermediate" | "advanced"
  tags: string[]
}

interface EducationalContent {
  title: string
  overview: string
  keyPoints: string[]
  examples: string[]
  mathematics?: string
  realWorld: string[]
  commonMisconceptions: string[]
  advanced?: {
    concepts: string[]
    implementations: string[]
    research: string[]
  }
}

const comprehensiveFAQs: FAQ[] = [
  // Beginner FAQs
  {
    question: "What exactly is a token and why isn't it just a word?",
    answer: "A token is the smallest unit that an LLM processes, but it's not the same as a word. Tokens are created using algorithms like BPE (Byte Pair Encoding) that balance efficiency and meaning. Common words like 'the' become single tokens, while rare words like 'unhappiness' might be split into ['un', 'happy', 'ness']. This allows models to handle any text, even words they've never seen before, by breaking them into familiar parts.",
    category: "beginner",
    tags: ["tokenization", "basics"]
  },
  {
    question: "How do language models understand meaning if they only see numbers?",
    answer: "LLMs don't 'understand' meaning the way humans do. Instead, they learn statistical patterns in how tokens are used together. Through training on massive text datasets, they learn that certain tokens appear in similar contexts, creating mathematical representations (embeddings) that capture semantic relationships. When tokens have similar embeddings, the model treats them as related concepts.",
    category: "beginner",
    tags: ["embeddings", "meaning", "understanding"]
  },
  {
    question: "What is attention and why is it so important?",
    answer: "Attention is the mechanism that allows the model to focus on relevant parts of the input when processing each token. Think of it like highlighting important words when reading. When processing 'The cat sat on the mat', the word 'sat' might pay strong attention to 'cat' (the subject) and 'mat' (where the sitting happens). This allows the model to understand relationships between words regardless of their distance in the text.",
    category: "beginner",
    tags: ["attention", "mechanism"]
  },
  {
    question: "How does ChatGPT generate text step by step?",
    answer: "ChatGPT generates text one token at a time in an autoregressive manner: 1) It processes your input through all its layers, 2) Predicts the most likely next token, 3) Adds that token to the sequence, 4) Uses the new sequence to predict the next token, 5) Repeats until it generates a complete response. Each prediction considers the entire context built so far.",
    category: "beginner",
    tags: ["generation", "autoregressive"]
  },
  {
    question: "Why do models sometimes give different answers to the same question?",
    answer: "This happens because of 'sampling' during generation. Instead of always picking the most likely next token (which would make responses predictable and repetitive), models use randomness controlled by parameters like 'temperature'. Higher temperature = more creative/random, lower temperature = more focused/deterministic. This randomness allows for diverse, natural-sounding responses.",
    category: "beginner",
    tags: ["sampling", "temperature", "randomness"]
  },

  // Intermediate FAQs
  {
    question: "What are the different types of attention and when are they used?",
    answer: "There are several types: 1) Self-attention: tokens attend to other tokens in the same sequence (used in all transformer layers), 2) Cross-attention: tokens from one sequence attend to another (used in encoder-decoder models for translation), 3) Causal/masked attention: prevents future-peeking in autoregressive models like GPT, 4) Multi-head attention: runs multiple attention mechanisms in parallel to capture different relationship types.",
    category: "intermediate",
    tags: ["attention", "types", "architecture"]
  },
  {
    question: "How do positional encodings work and why are they needed?",
    answer: "Transformers process all tokens simultaneously, unlike RNNs that process sequentially. Without positional information, 'cat chases dog' would be identical to 'dog chases cat'. Positional encodings add position information to each token's embedding. Methods include: 1) Sinusoidal encodings (original transformer), 2) Learned absolute positions, 3) Relative position encodings, 4) Rotary Position Embedding (RoPE) used in modern models.",
    category: "intermediate",
    tags: ["position", "encoding", "sequence"]
  },
  {
    question: "What happens in each layer of a transformer?",
    answer: "Each transformer layer has two main components: 1) Multi-head self-attention that lets tokens gather information from other tokens, 2) Feed-forward network that processes each token's representation independently. Both use residual connections and layer normalization. Early layers focus on syntax and local patterns, middle layers handle semantics and relationships, later layers perform task-specific reasoning.",
    category: "intermediate",
    tags: ["layers", "architecture", "processing"]
  },
  {
    question: "How do different model architectures (GPT vs BERT vs T5) differ?",
    answer: "GPT (decoder-only): Uses causal attention, generates text left-to-right, optimized for generation. BERT (encoder-only): Uses bidirectional attention, sees full context, optimized for understanding tasks. T5 (encoder-decoder): Encoder processes input bidirectionally, decoder generates output causally, good for text-to-text tasks like translation or summarization.",
    category: "intermediate",
    tags: ["architecture", "models", "comparison"]
  },
  {
    question: "What are embeddings and how do they capture meaning?",
    answer: "Embeddings are dense vector representations of tokens, typically 768-12,288 dimensions. They're learned during training so that tokens used in similar contexts have similar vectors. This creates a geometric space where semantic relationships become mathematical relationships: vector('king') - vector('man') + vector('woman') ≈ vector('queen'). The model performs computations in this embedding space.",
    category: "intermediate",
    tags: ["embeddings", "vectors", "semantics"]
  },

  // Advanced FAQs
  {
    question: "How do modern efficiency improvements like linear attention work?",
    answer: "Standard attention has O(n²) complexity due to computing all pairwise token interactions. Linear attention variants reduce this: 1) Linformer projects keys/values to lower dimensions, 2) Performer uses random feature maps to approximate attention, 3) Linear attention decomposes attention into smaller operations. These maintain most performance while scaling better to long sequences.",
    category: "advanced",
    tags: ["efficiency", "linear attention", "complexity"]
  },
  {
    question: "What are the different attention patterns that emerge in trained models?",
    answer: "Research has identified several attention patterns: 1) Local patterns (attending to nearby tokens), 2) Broadcast patterns (one token attending to many), 3) Syntactic patterns (following grammatical relationships), 4) Coreference patterns (connecting pronouns to antecedents), 5) Semantic patterns (connecting semantically related tokens). Different heads specialize in different patterns.",
    category: "advanced",
    tags: ["attention patterns", "interpretability", "specialization"]
  },
  {
    question: "How do scaling laws predict model capabilities?",
    answer: "Scaling laws show that model performance improves predictably with: 1) Model size (number of parameters), 2) Dataset size (number of training tokens), 3) Compute budget (FLOPs for training). These follow power law relationships. Capabilities like few-shot learning and reasoning emerge at certain scales, suggesting that scale alone can unlock new abilities.",
    category: "advanced",
    tags: ["scaling laws", "capabilities", "emergence"]
  },
  {
    question: "What are the current research frontiers in transformer architectures?",
    answer: "Key research areas include: 1) Efficiency improvements (sparse attention, linear complexity), 2) Long context handling (extending to millions of tokens), 3) Multimodal integration (combining text, vision, audio), 4) Interpretability (understanding what models learn), 5) Specialized architectures (mixture of experts, retrieval-augmented generation), 6) Memory mechanisms (persistent memory across interactions).",
    category: "advanced",
    tags: ["research", "frontiers", "future"]
  },
  {
    question: "How do in-context learning and few-shot prompting work mechanistically?",
    answer: "In-context learning allows models to adapt to new tasks using just examples in the prompt, without parameter updates. Mechanistically, this likely works through: 1) Pattern matching in attention layers, 2) Task identification from examples, 3) Analogy-making between examples and the new instance, 4) Internal activation of 'task vectors' that guide prediction. The model learns to recognize and apply patterns from the context.",
    category: "advanced",
    tags: ["in-context learning", "few-shot", "mechanisms"]
  }
]

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
    ],
    advanced: {
      concepts: [
        "Positional encodings add sequence information to token embeddings",
        "Layer-wise representation learning shows how embeddings evolve through the network",
        "Subword tokenization allows handling of out-of-vocabulary words"
      ],
      implementations: [
        "Sinusoidal vs learned positional encodings trade-offs",
        "RoPE (Rotary Position Embedding) rotates vectors in complex space",
        "Embedding tying shares parameters between input and output embeddings"
      ],
      research: [
        "Probing studies reveal what linguistic information embeddings capture",
        "Cross-lingual embeddings enable zero-shot transfer between languages",
        "Contextual embeddings change based on surrounding tokens"
      ]
    }
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
    ],
    advanced: {
      concepts: [
        "Multi-head attention runs multiple attention mechanisms in parallel",
        "Causal masking prevents future information leakage in autoregressive models",
        "Cross-attention connects encoder and decoder in sequence-to-sequence tasks"
      ],
      implementations: [
        "Linear attention variants reduce O(n²) complexity to O(n)",
        "Sparse attention patterns like Longformer's sliding window + global tokens",
        "Flash Attention optimizes memory usage for long sequences"
      ],
      research: [
        "Attention head specialization studies reveal different linguistic functions",
        "Attention entropy measures focus vs dispersion of attention patterns",
        "Gradient-based attention analysis shows information flow through layers"
      ]
    }
  },

  positional: {
    title: "Positional Encodings: Teaching Order to Parallel Processing",
    overview: "Since transformers process all tokens simultaneously, positional encodings add crucial sequence order information that distinguishes 'cat chases dog' from 'dog chases cat'.",
    keyPoints: [
      "Transformers have no inherent notion of sequence order",
      "Sinusoidal encodings use mathematical functions for position",
      "Learned positional encodings are trainable parameters",
      "Relative position encodings focus on token distances",
      "RoPE (Rotary Position Embedding) rotates vectors in complex space"
    ],
    examples: [
      'Absolute: Position 1 gets encoding [0.1, -0.2, 0.4, ...]',
      'Relative: "cat" to "sat" = distance +2 positions',
      'RoPE: Rotates query/key vectors based on position',
      'Impact: Without position, "The cat sat" = "sat cat The"'
    ],
    mathematics: "PE(pos,2i) = sin(pos/10000^(2i/d_model)), PE(pos,2i+1) = cos(pos/10000^(2i/d_model))",
    realWorld: [
      "Different models use different positional encoding strategies",
      "Position encoding affects how well models handle long sequences",
      "RoPE is becoming standard in modern large language models",
      "Position interpolation allows handling longer sequences than training"
    ],
    commonMisconceptions: [
      "❌ Position encodings are not just added once - they affect all layers",
      "❌ Relative position is not always better than absolute position",
      "❌ Position encodings don't just mark sequence order - they enable contextual understanding",
      "❌ Learned positions aren't necessarily better than sinusoidal encodings"
    ]
  },

  layers: {
    title: "Multi-Layered Processing: How Understanding Emerges Through Depth",
    overview: "Each transformer layer refines token representations, with early layers handling syntax and local patterns, middle layers processing semantics and relationships, and later layers performing complex reasoning.",
    keyPoints: [
      "Early layers (1-4): Syntax, POS tagging, local dependencies",
      "Middle layers (5-8): Semantics, entity relationships, co-reference",
      "Later layers (9-12): Abstract reasoning, task-specific adaptations",
      "Each layer has attention + feed-forward + residual connections",
      "Layer normalization stabilizes training in deep networks"
    ],
    examples: [
      'Layer 1: "cat" → [basic noun features, local context]',
      'Layer 6: "cat" → [animal, subject of sentence, specific referent]',
      'Layer 12: "cat" → [story character, narrative element, prediction context]',
      'Residual connections: Enable gradient flow through 96+ layers'
    ],
    mathematics: "layer_out = LayerNorm(x + MultiHeadAttention(x)) + LayerNorm(x + FFN(x))",
    realWorld: [
      "GPT-4 has 96 layers, each with specific learned functions",
      "Layer representations can be probed to understand what models learn",
      "Different layers activate for different types of tasks",
      "Deeper models generally perform better on complex reasoning"
    ],
    commonMisconceptions: [
      "❌ More layers don't always mean better performance",
      "❌ Layer functions aren't manually programmed - they emerge from training",
      "❌ Information doesn't flow linearly - residual connections create shortcuts",
      "❌ Early layers aren't 'simpler' - they handle foundational processing"
    ]
  },

  generation: {
    title: "Text Generation: From Probability Distributions to Language",
    overview: "Language models generate text by predicting probability distributions over the vocabulary for each next token, using various sampling strategies to control randomness and creativity.",
    keyPoints: [
      "Autoregressive generation: predict one token at a time",
      "Temperature controls randomness (0.0 = deterministic, 2.0 = very random)",
      "Top-k sampling considers only k most likely tokens",
      "Top-p (nucleus) sampling uses cumulative probability threshold",
      "Beam search explores multiple generation paths simultaneously"
    ],
    examples: [
      'Input: "The weather is" → Probabilities: [sunny: 0.3, nice: 0.2, ...]',
      'Temperature 0.1: Always picks "sunny" (deterministic)',
      'Temperature 1.0: Samples randomly based on probabilities',
      'Top-k=10: Only consider 10 most likely next tokens'
    ],
    mathematics: "P(token) = softmax(logits / temperature), next_token = sample(P)",
    realWorld: [
      "ChatGPT uses sophisticated sampling with multiple parameters",
      "Different tasks benefit from different generation strategies",
      "Repetition penalty prevents endless loops in generation",
      "Length penalties encourage appropriate response lengths"
    ],
    commonMisconceptions: [
      "❌ Models don't plan entire responses - they generate word by word",
      "❌ Higher temperature doesn't always mean better creativity",
      "❌ Models don't have consciousness - they follow statistical patterns",
      "❌ Deterministic generation (temp=0) isn't always optimal"
    ]
  },

  limitations: {
    title: "Understanding Model Limitations and Responsible Use",
    overview: "While powerful, LLMs have significant limitations including training data cutoffs, hallucination, bias, and lack of true understanding. Recognizing these is crucial for responsible deployment.",
    keyPoints: [
      "Training data cutoff creates knowledge gaps",
      "Hallucination: confident-sounding but incorrect information",
      "Biases reflect training data patterns and societal biases",
      "No true understanding - pattern matching, not reasoning",
      "Context window limits affect long document processing"
    ],
    examples: [
      'Hallucination: Making up fake citations or facts',
      'Bias: Stereotypical associations (doctor=male, nurse=female)',
      'Knowledge cutoff: Missing recent events after training',
      'Context limits: Forgetting start of very long conversations'
    ],
    mathematics: "bias_score = E[model_output | demographic_attribute] - E[model_output]",
    realWorld: [
      "Models can perpetuate harmful stereotypes and misinformation",
      "Fact-checking and human oversight remain essential",
      "Different models have different biases and capabilities",
      "Regular evaluation and updates help address limitations"
    ],
    commonMisconceptions: [
      "❌ LLMs are not omniscient - they have clear knowledge boundaries",
      "❌ High confidence doesn't indicate accuracy",
      "❌ Models don't have intentions or consciousness",
      "❌ Bigger models don't automatically solve bias problems"
    ]
  }
}

interface EducationalDocsProps {
  activeSection?: string
}

export function EducationalDocs({ activeSection = "tokens" }: EducationalDocsProps) {
  const [selectedSection, setSelectedSection] = useState(activeSection)
  const [showMath, setShowMath] = useState(false)
  const [selectedFAQCategory, setSelectedFAQCategory] = useState<"all" | "beginner" | "intermediate" | "advanced">("all")
  const [expandedFAQs, setExpandedFAQs] = useState<Set<number>>(new Set())

  const sections = Object.keys(educationalSections)
  const currentContent = educationalSections[selectedSection]

  const filteredFAQs = selectedFAQCategory === "all" 
    ? comprehensiveFAQs 
    : comprehensiveFAQs.filter(faq => faq.category === selectedFAQCategory)

  const toggleFAQ = (index: number) => {
    const newExpanded = new Set(expandedFAQs)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedFAQs(newExpanded)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <BookOpen className="h-6 w-6" />
              <span>Comprehensive LLM Educational Documentation</span>
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
              <Badge variant="outline">Enhanced Guide</Badge>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main Navigation Tabs */}
      <Tabs defaultValue="concepts" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="concepts" className="flex items-center space-x-2">
            <Brain className="h-4 w-4" />
            <span>Core Concepts</span>
          </TabsTrigger>
          <TabsTrigger value="multihead" className="flex items-center space-x-2">
            <Network className="h-4 w-4" />
            <span>Multi-Head Attention</span>
          </TabsTrigger>
          <TabsTrigger value="advanced" className="flex items-center space-x-2">
            <Microscope className="h-4 w-4" />
            <span>Advanced Topics</span>
          </TabsTrigger>
          <TabsTrigger value="faq" className="flex items-center space-x-2">
            <HelpCircle className="h-4 w-4" />
            <span>Comprehensive FAQ</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="multihead" className="space-y-6">
          <MultiHeadAttentionVisualizer />
        </TabsContent>

        <TabsContent value="concepts" className="space-y-6">
          {/* Section Navigation */}
          <Card>
            <CardContent className="p-6">
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                {[
                  { key: "tokens", icon: Zap, title: "Tokenization", desc: "Text to tokens" },
                  { key: "embeddings", icon: Brain, title: "Embeddings", desc: "Vectors & meaning" },
                  { key: "attention", icon: Network, title: "Attention", desc: "QKV mechanism" },
                  { key: "positional", icon: Target, title: "Position", desc: "Sequence order" },
                  { key: "layers", icon: Layers, title: "Layer Processing", desc: "Multi-layer flow" },
                  { key: "generation", icon: Settings, title: "Generation", desc: "Text creation" }
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
                    <div className="flex flex-col items-center space-y-2">
                      <Icon className="h-5 w-5 text-blue-600" />
                      <div className="text-center">
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
                      {selectedSection === "positional" && "Positional encodings solve the fundamental problem that transformers process tokens in parallel rather than sequentially. Without them, models couldn't distinguish between different orderings of the same words."}
                      {selectedSection === "layers" && "Understanding layer-by-layer processing reveals how complex understanding emerges from simple mathematical operations repeated many times. This hierarchical processing is key to transformer capabilities."}
                      {selectedSection === "generation" && "Text generation techniques control the creativity, consistency, and quality of model outputs. Understanding these parameters is essential for getting desired behavior from language models."}
                    </p>
                  </div>

                  {/* Advanced Concepts (if available) */}
                  {currentContent.advanced && (
                    <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                      <h3 className="font-medium text-purple-800 dark:text-purple-200 mb-3">Advanced Concepts</h3>
                      <div className="space-y-3">
                        <div>
                          <h4 className="font-medium text-sm mb-1">Key Concepts:</h4>
                          <ul className="text-xs space-y-1">
                            {currentContent.advanced.concepts.map((concept, index) => (
                              <li key={index} className="text-purple-700 dark:text-purple-300">• {concept}</li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <h4 className="font-medium text-sm mb-1">Implementation Details:</h4>
                          <ul className="text-xs space-y-1">
                            {currentContent.advanced.implementations.map((impl, index) => (
                              <li key={index} className="text-purple-700 dark:text-purple-300">• {impl}</li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <h4 className="font-medium text-sm mb-1">Research Frontiers:</h4>
                          <ul className="text-xs space-y-1">
                            {currentContent.advanced.research.map((research, index) => (
                              <li key={index} className="text-purple-700 dark:text-purple-300">• {research}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  )}
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
                          {selectedSection === "positional" && (
                            <div>
                              <p><strong>Sinusoidal Encoding:</strong></p>
                              <p>Uses sine and cosine functions with different frequencies to create unique position signatures that the model can learn to interpret.</p>
                            </div>
                          )}
                          {selectedSection === "layers" && (
                            <div>
                              <p><strong>Transformer Layer Components:</strong></p>
                              <p>Each layer applies multi-head attention and feed-forward networks with residual connections and layer normalization for stable training.</p>
                            </div>
                          )}
                          {selectedSection === "generation" && (
                            <div>
                              <p><strong>Sampling Process:</strong></p>
                              <p>Temperature controls the sharpness of the probability distribution. Lower values make the model more confident, higher values increase randomness.</p>
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
        </TabsContent>

        <TabsContent value="advanced" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Microscope className="h-6 w-6" />
                <span>Advanced Topics & Research Frontiers</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="font-semibold text-lg">Efficiency Improvements</h3>
                  <div className="space-y-3">
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <h4 className="font-medium text-sm mb-1">Linear Attention</h4>
                      <p className="text-xs text-gray-600 dark:text-gray-300">Reduces O(n²) complexity to O(n) for long sequences</p>
                    </div>
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <h4 className="font-medium text-sm mb-1">Sparse Attention</h4>
                      <p className="text-xs text-gray-600 dark:text-gray-300">Local + global patterns (Longformer, BigBird)</p>
                    </div>
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <h4 className="font-medium text-sm mb-1">Flash Attention</h4>
                      <p className="text-xs text-gray-600 dark:text-gray-300">Memory-efficient attention computation</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h3 className="font-semibold text-lg">Architectural Innovations</h3>
                  <div className="space-y-3">
                    <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <h4 className="font-medium text-sm mb-1">Mixture of Experts</h4>
                      <p className="text-xs text-gray-600 dark:text-gray-300">Conditional computation with specialized sub-networks</p>
                    </div>
                    <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <h4 className="font-medium text-sm mb-1">Retrieval Augmentation</h4>
                      <p className="text-xs text-gray-600 dark:text-gray-300">External memory and knowledge base integration</p>
                    </div>
                    <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <h4 className="font-medium text-sm mb-1">Cross-Modal</h4>
                      <p className="text-xs text-gray-600 dark:text-gray-300">Vision-language models (CLIP, GPT-4V)</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h3 className="font-semibold text-lg">Interpretability</h3>
                  <div className="space-y-3">
                    <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                      <h4 className="font-medium text-sm mb-1">Mechanistic Interpretability</h4>
                      <p className="text-xs text-gray-600 dark:text-gray-300">Understanding circuits and features learned by models</p>
                    </div>
                    <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                      <h4 className="font-medium text-sm mb-1">Probing Studies</h4>
                      <p className="text-xs text-gray-600 dark:text-gray-300">What linguistic knowledge is captured in representations</p>
                    </div>
                    <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                      <h4 className="font-medium text-sm mb-1">Attention Analysis</h4>
                      <p className="text-xs text-gray-600 dark:text-gray-300">Head specialization and pattern interpretation</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h3 className="font-semibold text-lg">Scaling & Emergence</h3>
                  <div className="space-y-3">
                    <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                      <h4 className="font-medium text-sm mb-1">Scaling Laws</h4>
                      <p className="text-xs text-gray-600 dark:text-gray-300">Predictable performance improvements with scale</p>
                    </div>
                    <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                      <h4 className="font-medium text-sm mb-1">Emergent Abilities</h4>
                      <p className="text-xs text-gray-600 dark:text-gray-300">New capabilities appearing at certain scales</p>
                    </div>
                    <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                      <h4 className="font-medium text-sm mb-1">In-Context Learning</h4>
                      <p className="text-xs text-gray-600 dark:text-gray-300">Learning from examples in the prompt</p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="faq" className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center space-x-2">
                  <HelpCircle className="h-6 w-6" />
                  <span>Comprehensive FAQ</span>
                </CardTitle>
                <div className="flex items-center space-x-2">
                  <Button
                    variant={selectedFAQCategory === "all" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSelectedFAQCategory("all")}
                  >
                    All ({comprehensiveFAQs.length})
                  </Button>
                  <Button
                    variant={selectedFAQCategory === "beginner" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSelectedFAQCategory("beginner")}
                  >
                    Beginner ({comprehensiveFAQs.filter(f => f.category === "beginner").length})
                  </Button>
                  <Button
                    variant={selectedFAQCategory === "intermediate" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSelectedFAQCategory("intermediate")}
                  >
                    Intermediate ({comprehensiveFAQs.filter(f => f.category === "intermediate").length})
                  </Button>
                  <Button
                    variant={selectedFAQCategory === "advanced" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSelectedFAQCategory("advanced")}
                  >
                    Advanced ({comprehensiveFAQs.filter(f => f.category === "advanced").length})
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {filteredFAQs.map((faq, index) => (
                  <Collapsible 
                    key={index}
                    open={expandedFAQs.has(index)}
                    onOpenChange={() => toggleFAQ(index)}
                  >
                    <CollapsibleTrigger asChild>
                      <Button
                        variant="ghost"
                        className="w-full justify-between p-4 h-auto text-left hover:bg-gray-50 dark:hover:bg-gray-800"
                      >
                        <div className="flex items-start space-x-3">
                          <Badge 
                            variant={faq.category === "beginner" ? "default" : faq.category === "intermediate" ? "secondary" : "destructive"}
                            className="mt-1"
                          >
                            {faq.category}
                          </Badge>
                          <div className="flex-1">
                            <h3 className="font-medium text-sm">{faq.question}</h3>
                            <div className="flex items-center space-x-1 mt-1">
                              {faq.tags.map((tag, tagIndex) => (
                                <Badge key={tagIndex} variant="outline" className="text-xs">
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        </div>
                        {expandedFAQs.has(index) ? (
                          <ChevronDown className="h-4 w-4" />
                        ) : (
                          <ChevronRight className="h-4 w-4" />
                        )}
                      </Button>
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-4">
                      <div className="pl-12">
                        <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                          {faq.answer}
                        </p>
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Quick Reference */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Info className="h-5 w-5" />
            <span>Quick Reference - Model Specifications</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
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
            <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
              <div className="font-medium text-sm text-red-800 dark:text-red-200">Parameters</div>
              <div className="text-xs text-red-600 dark:text-red-300">GPT-4: ~1.76T parameters</div>
            </div>
            <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded">
              <div className="font-medium text-sm text-yellow-800 dark:text-yellow-200">Context</div>
              <div className="text-xs text-yellow-600 dark:text-yellow-300">GPT-4: 128K tokens</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
