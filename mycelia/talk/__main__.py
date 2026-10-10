"""Talk to Mycelia.

  python -m mycelia.talk            command chat (no AI model, answers from real data)
  python -m mycelia.talk --claude   natural language through Claude (needs ANTHROPIC_API_KEY
                                    or an `ant auth login` profile; Claude can only use Mycelia's
                                    commands as tools)
"""
import argparse
from .commands import Mycelia, HELP


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--claude', action='store_true'); ap.add_argument('--bundle', default='lab_state/talk/mycelia-bundle.pkl')
    ap.add_argument('--lab', default='lab_state'); ap.add_argument('--effort', default='medium')
    a = ap.parse_args(); m = Mycelia(a.bundle, a.lab)
    chat = None
    if a.claude:
        from .llm import ClaudeChat
        chat = ClaudeChat(m, effort=a.effort)
        print('Mycelia (via Claude). Ask in plain English; "quit" to exit. Research prototype, not medical advice.')
    else:
        print('Mycelia command chat. Research prototype, not medical advice.\n' + HELP)
    while True:
        try:
            line = input('you> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ('quit', 'exit'):
            break
        if line:
            print('mycelia> ' + (chat.send(line) if chat else m.run(line)))


if __name__ == '__main__':
    main()
