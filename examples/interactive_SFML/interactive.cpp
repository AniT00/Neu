#include "NeuralNetwork.h"
#include "SFML/Graphics.hpp"

#define WINDOW_WIDTH	720
#define WINDOW_HEIGTH	480
#define RED_OFF			0
#define GREEN_OFF		1
#define BLUE_OFF		2
#define ALPHA_OFF		3


// Color offset of positive network output.
#define POS_OUTPUT_COLOR	BLUE_OFF
// Color offset of negative network output.
#define NEG_OUTPUT_COLOR	GREEN_OFF
// Max intensity of colors when visualizing perceptron work.
#define MAX_INTENSITY	200

// Unit of lenth in pixels.
#define SCALE			100

// Network configuration.
constexpr std::initializer_list<size_t> configuration = { 2, 3, 3, 1 };

int main()
{
	sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGTH), "window");

	NeuralNetwork network(configuration);
	size_t sampleSize = 4;
	std::vector<sf::Vector2f> coords;
	std::vector<float> answers;
	std::vector<sf::CircleShape> points;
	
	network.setLogger(nullptr)
		.setInputSample((float*)coords.data(), sampleSize)
		.setBatchSize(1)
		.setEpochs(10)
		.setLearningRate(.7)
    .setLossFunction(LossFunction::CrossEntropy)
		.setAnswers(answers.data());

	sf::Texture texture;
	texture.create(WINDOW_WIDTH, WINDOW_HEIGTH);

	sf::Uint8* pixels = new sf::Uint8[WINDOW_WIDTH * WINDOW_HEIGTH * 4]{ 0 };

	for (size_t i = 0; i < WINDOW_WIDTH * WINDOW_HEIGTH; i++)
	{
		pixels[i * 4 + ALPHA_OFF] = 255;
	}
	texture.update(pixels);
	sf::Sprite sp;
	sp.setTexture(texture);

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}
			if (event.type == sf::Event::MouseButtonPressed) {
				if (event.key.code == sf::Mouse::Left ||
					event.key.code == sf::Mouse::Right) {
					sf::CircleShape r(5.f);
					r.setOrigin(5.f, 5.f);
					sf::Vector2i pos = sf::Mouse::getPosition(window);
					r.setPosition(pos.x, pos.y);
					r.setFillColor(event.key.code == sf::Mouse::Left ? sf::Color::Blue : sf::Color::Green);
					points.push_back(r);
					answers.push_back(event.key.code == sf::Mouse::Left ? 1 : 0);
					coords.push_back(sf::Vector2f((float)pos.x / SCALE, (float)pos.y / SCALE));
					network
						.setInputSample((float*)coords.data(), coords.size())
						.setAnswers(answers.data());
				}
			}
		}

		if (!points.empty()) {
			network.train();
			for (size_t i = 0; i < WINDOW_WIDTH * WINDOW_HEIGTH; i++)
			{
				sf::Vector2f point((float)(i % WINDOW_WIDTH) / SCALE, (float)i / WINDOW_WIDTH / SCALE);
				const float* result = network.predict((float*)&point);
				pixels[i * 4 + NEG_OUTPUT_COLOR] = MAX_INTENSITY * (1 - result[0]);
				pixels[i * 4 + POS_OUTPUT_COLOR] = result[0] * MAX_INTENSITY;
			}
			texture.update(pixels);
		}

		window.clear(sf::Color::White);
		window.draw(sp);
		std::for_each(points.begin(), points.end(), [&](auto p) { window.draw(p); });
		window.display();
	}
	return 0;
}
