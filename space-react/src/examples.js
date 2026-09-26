const EMOTIONS = [
  "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
  "curiosity", "disappointment", "disapproval", "excitement", "fear", "gratitude", "joy",
  "love", "optimism", "realization", "sadness", "surprise", "neutral",
];

const BANKING = [
  "Unable to verify identity", "Activate my card", "Age limit", "Apple pay or google pay",
  "Card arrival", "Card not working", "Change pin", "Lost or stolen card",
  "Pending card payment", "Top up by card", "Verify my identity", "Why verify identity",
];

const VOICE = [
  "Alarm set", "Calendar query", "Cooking recipe", "Datetime query", "Email send",
  "Music play", "News query", "Recommendation events", "Recommendation locations",
  "Transport query", "Weather query",
];

const LEGAL = [
  "Adjustments", "Amendments", "Arbitration", "Assignments", "Confidentiality",
  "Counterparts", "Expenses", "Governing Laws", "Indemnifications", "Notices",
  "Severability", "Terminations", "Waivers", "Warranties",
];

export const EXAMPLES = [
  {
    label: "Bank support",
    hint: "77 real categories, trimmed to 12 here",
    background: "How long will it take for my ID to verify?",
    question: "Which banking support intent does the customer's message express?",
    options: BANKING,
  },
  {
    label: "Emotion",
    hint: "a comment from Reddit",
    background:
      "I did that in origins but now that theres mercenaries and its sooo annoying to deal " +
      "with them so i stay as quiet as i can until im caught.",
    question: "Which emotion does the comment primarily express?",
    options: EMOTIONS,
  },
  {
    label: "Voice assistant",
    hint: "what is the user asking for?",
    background: "are there any free events on in my area today",
    question: "Which intent does the user's request to a voice assistant express?",
    options: VOICE,
  },
  {
    label: "Legal clause",
    hint: "a real contract paragraph",
    background:
      "Any notices or demands required or contemplated hereunder shall be written and shall " +
      "be effective two days after the placing thereof in the United States mails postage " +
      "prepaid or with a nationally-recognized courier service such as Federal Express, " +
      "addressed to the relevant party at its address set forth below.",
    question: "Which contract provision category does this clause belong to?",
    options: LEGAL,
  },
  {
    label: "Sentence logic",
    hint: "structured input, given as JSON",
    background: JSON.stringify(
      {
        premise: "All the customers received their refunds yesterday.",
        hypothesis: "Some customers have not received their refunds.",
      },
      null,
      2,
    ),
    question: "What is the relationship of `hypothesis` to `premise`?",
    options: [
      "The hypothesis is definitely true given the premise.",
      "The hypothesis might be true; the premise does not settle it.",
      "The hypothesis is definitely false given the premise.",
    ],
  },
];
