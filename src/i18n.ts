import { i18n } from '@nekazari/sdk';
import en from './locales/en.json';
import es from './locales/es.json';
import eu from './locales/eu.json';
import fr from './locales/fr.json';
import pt from './locales/pt.json';
import ca from './locales/ca.json';

const NS = 'weather-map';

function register(): void {
  const add = i18n && typeof i18n.addResourceBundle === 'function' ? i18n.addResourceBundle : undefined;
  if (!add) return;
  for (const [lang, resources] of Object.entries({ en, es, eu, fr, pt, ca })) {
    add.call(i18n, lang, NS, resources, true, true);
  }
}

register();
