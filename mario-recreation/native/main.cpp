#define SDL_MAIN_HANDLED
#include "Camera.h"
#include "Constants.h"
#include "Input.h"
#include "Random.h"
#include "SoundManager.h"
#include "TextureManager.h"
#include "command/CommandScheduler.h"
#include "scenes/GameScene.h"
#include "systems/ScoreSystem.h"
#include "systems/PlayerSystem.h"
#include <SDL_mixer.h>
#include <array>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
constexpr Key buttons[] = {Key::RIGHT, Key::LEFT, Key::JUMP, Key::SPRINT, Key::DUCK, Key::FIREBALL};
void input(unsigned mask) {
   std::array<Uint8, SDL_NUM_SCANCODES> keys{};
   for (unsigned i = 0; i < 6; ++i)
      if (mask & (1u << i)) keys[Input::Get().getBoundKey(buttons[i])] = 1;
   Input::Get().update(keys.data());
}
void tick(GameScene& scene, unsigned mask) {
   input(mask);
   scene.handleInput();
   scene.update();
   CommandScheduler::getInstance().run();
}
bool dead(GameScene& scene) {
   return scene.getWorld()->getSystem<PlayerSystem>()->currentState == PlayerSystem::GAMEOVER;
}
void state(GameScene& scene, unsigned long long frame, bool terminal) {
   World* world = scene.getWorld();
   auto* score = world->getSystem<ScoreSystem>();
   auto* p = world->findFirst<PlayerComponent, PositionComponent>();
   auto* pos = p->getComponent<PositionComponent>();
   auto* move = p->getComponent<MovingComponent>();
   std::cout << std::setprecision(9) << std::boolalpha;
   std::cout << "{\"frame\":" << frame << ",\"level\":\"" << scene.getLevel() << '-' << scene.getSublevel()
      << "\",\"camera\":[" << Camera::Get().getCameraX()/2 << ',' << Camera::Get().getCameraY()/2
      << "],\"score\":" << score->getScore() << ",\"coins\":" << score->getCoins()
      << ",\"lives\":" << score->getLives() << ",\"time\":" << score->getGameTime()
      << ",\"terminated\":" << terminal << ",\"dead\":" << dead(scene)
      << ",\"win\":" << scene.gameWin() << ",\"controllable\":" << PlayerSystem::isInputEnabled()
      << ",\"transition\":" << world->getSystem<RenderSystem>()->isTransitionRendering()
      << ",\"player\":{\"x\":" << pos->position.x/2 << ",\"y\":" << pos->position.y/2
      << ",\"vx\":" << move->velocity.x/2 << ",\"vy\":" << move->velocity.y/2
      << ",\"width\":" << pos->scale.x/2 << ",\"height\":" << pos->scale.y/2
      << ",\"power\":" << int(p->getComponent<PlayerComponent>()->playerState)
      << ",\"star\":" << p->getComponent<PlayerComponent>()->superStar
      << ",\"animation\":" << int(world->getSystem<PlayerSystem>()->currentState)
      << "},\"objects\":[";
   bool first = true;
   for (auto* e : world->getEntities()) {
      if (!e->hasComponent<PositionComponent>() || e == p) continue;
      const char* kind = e->hasComponent<EnemyComponent>() ? "enemy" :
         e->hasComponent<CollectibleComponent>() ? "item" :
         e->hasComponent<ProjectileComponent>() ? "projectile" :
         e->hasComponent<TileComponent>() ? "solid" : nullptr;
      if (!kind) continue;
      auto* q = e->getComponent<PositionComponent>();
      if (!Camera::Get().inCameraRange(q)) continue;
      if (!first) std::cout << ',';
      first = false;
      std::cout << "{\"kind\":\"" << kind << "\",\"x\":" << q->position.x/2
         << ",\"y\":" << q->position.y/2 << ",\"width\":" << q->scale.x/2
         << ",\"height\":" << q->scale.y/2;
      if (e->hasComponent<EnemyComponent>())
         std::cout << ",\"type\":" << int(e->getComponent<EnemyComponent>()->enemyType);
      if (e->hasComponent<MovingComponent>()) {
         auto* m = e->getComponent<MovingComponent>();
         std::cout << ",\"vx\":" << m->velocity.x/2 << ",\"vy\":" << m->velocity.y/2;
      }
      std::cout << '}';
   }
   std::cout << "]}" << std::endl;
}
}

int main(int argc, char** argv) {
   try {
      bool headless = false, muted = false;
      int level = 1, stage = 1;
      unsigned seed = 0;
      for (int i = 1; i < argc; ++i) {
         const std::string arg = argv[i];
         if (arg == "--headless") headless = true;
         else if (arg == "--mute") muted = true;
         else if (arg == "--seed" && i+1 < argc) seed = std::stoul(argv[++i]);
         else if (arg == "--level" && i+1 < argc) {
            std::string value = argv[++i];
            if (value.size()!=3 || value[1]!='-' || value[0]<'1' || value[0]>'8' || value[2]<'1' || value[2]>'4')
               throw std::runtime_error("Level must be 1-1 through 8-4");
            level = value[0]-'0'; stage = value[2]-'0';
         } else if (arg == "--help") {
            std::cout << "mario [--headless] [--mute] [--level 1-1] [--seed 0]\n"
               "Play: A/D or arrows move, Space/Z jump, Shift/X run+fire, S/down duck, Escape pause.\n"
               "Headless stdin: step MASK FRAMES | state | capture \"ABSOLUTE.png\" | quit\n";
            return 0;
         } else throw std::runtime_error("Unknown or incomplete argument: " + arg);
      }
      std::filesystem::current_path(MARIO_RESOURCE_DIR);
      if (headless) {
         SDL_setenv("MARIO_HEADLESS", "1", 1);
         SDL_setenv("SDL_VIDEODRIVER", "dummy", 1);
         SDL_setenv("SDL_AUDIODRIVER", "dummy", 1);
      }
      if (TextureManager::Get().Init() || SoundManager::Get().Init()) return 1;
      if (muted || headless) { Mix_Volume(-1, 0); Mix_VolumeMusic(0); }
      MarioRandom::seed(seed);
      {
         GameScene scene(level, stage);
         unsigned long long frame = 0;
         bool terminal = false;
         // Deterministic boot; frame zero is the first controllable gameplay observation.
         if (headless) {
            for (int i = 0; i < 180; ++i) tick(scene, 0);
            scene.getWorld()->getSystem<RenderSystem>()->tick(scene.getWorld());
            state(scene, frame, terminal);
            std::string line;
            while (std::getline(std::cin, line)) {
               std::istringstream in(line);
               std::string op; in >> op;
               if (op == "quit") break;
               if (op == "state") state(scene, frame, terminal);
               else if (op == "capture") {
                  std::string path; in >> std::quoted(path);
                  const bool ok = !path.empty() && TextureManager::Get().SaveFrame(path.c_str());
                  std::cout << (ok ? "{\"ok\":true}" : "{\"error\":\"capture failed\"}") << std::endl;
               } else if (op == "step") {
                  unsigned mask = 0; int frames = 0;
                  if (!(in >> mask >> frames) || mask > 63 || frames < 1 || frames > 3600) {
                     std::cout << "{\"error\":\"expected step MASK(0..63) FRAMES(1..3600)\"}" << std::endl;
                     continue;
                  }
                  for (int i = 0; i < frames && !terminal; ++i) {
                     tick(scene, mask); ++frame;
                     terminal = dead(scene) || scene.isFinished() || scene.gameWin();
                  }
                  // Scheduler can change a level after its last render. Render without ticking.
                  scene.getWorld()->getSystem<RenderSystem>()->tick(scene.getWorld());
                  state(scene, frame, terminal);
               } else std::cout << "{\"error\":\"unknown command\"}" << std::endl;
            }
         } else {
            bool running = true;
            const double frequency = double(SDL_GetPerformanceFrequency());
            double next = double(SDL_GetPerformanceCounter());
            while (running && !scene.isFinished()) {
               SDL_Event event;
               while (SDL_PollEvent(&event)) if (event.type == SDL_QUIT) running = false;
               const Uint8* raw = SDL_GetKeyboardState(nullptr);
               std::array<Uint8, SDL_NUM_SCANCODES> keys;
               std::copy(raw, raw + SDL_NUM_SCANCODES, keys.begin());
               keys[SDL_SCANCODE_A] |= raw[SDL_SCANCODE_LEFT];
               keys[SDL_SCANCODE_D] |= raw[SDL_SCANCODE_RIGHT];
               keys[SDL_SCANCODE_S] |= raw[SDL_SCANCODE_DOWN];
               keys[SDL_SCANCODE_SPACE] |= raw[SDL_SCANCODE_Z];
               keys[SDL_SCANCODE_LSHIFT] |= raw[SDL_SCANCODE_X];
               keys[SDL_SCANCODE_Q] |= keys[SDL_SCANCODE_LSHIFT];
               Input::Get().update(keys.data());
               scene.handleInput(); scene.update(); CommandScheduler::getInstance().run();
               next += frequency / 60.0;
               const double delay = (next - SDL_GetPerformanceCounter()) * 1000 / frequency;
               if (delay > 0) SDL_Delay(Uint32(delay));
               if (delay < -250) next = SDL_GetPerformanceCounter();
            }
         }
         CommandScheduler::getInstance().clear();
      }
      SoundManager::Get().Quit();
      TextureManager::Get().Quit();
      return 0;
   } catch (const std::exception& e) {
      std::cerr << "mario: " << e.what() << std::endl;
      return 1;
   }
}
