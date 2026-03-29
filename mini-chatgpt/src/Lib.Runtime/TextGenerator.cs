using MiniChatGPT.Contracts;
using MiniChatGPT.Sampling.Interfaces;
using Lib.MathCore;

namespace Lib.Runtime
{
    public class RuntimeTextGenerator : ITextGenerator
    {
        private readonly ILanguageModel _model;
        private readonly ISampler _sampler;
        private readonly IMathOps _mathOps;

        public RuntimeTextGenerator(ILanguageModel model, ISampler sampler, IMathOps mathOps)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _sampler = sampler ?? throw new ArgumentNullException(nameof(sampler));
            _mathOps = mathOps ?? throw new ArgumentNullException(nameof(mathOps));
        }

        public string Generate(string prompt, int maxTokens, float temperature, int topK, int? seed = null)
        {
            List<int> currentContext = new List<int> { Math.Abs(prompt.GetHashCode()) % _model.VocabSize };

            List<string> generatedWords = new List<string>();

            for (int i = 0; i < maxTokens; i++)
            {
                float[] logits = _model.NextTokenScores(currentContext.ToArray());

                float[] probabilities = _mathOps.Softmax(logits);

                int nextTokenId = _sampler.SampleWithSeed(probabilities, temperature, topK, seed);

                if (nextTokenId == 0)
                {
                    break;
                }

                currentContext.Add(nextTokenId);

                generatedWords.Add($"[Token_{nextTokenId}]");
            }

            return string.Join(" ", generatedWords);
        }
    }
}
