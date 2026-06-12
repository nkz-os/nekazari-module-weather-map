import en from './locales/en.json';
import es from './locales/es.json';
import eu from './locales/eu.json';
import fr from './locales/fr.json';
import pt from './locales/pt.json';
import ca from './locales/ca.json';

if (typeof window !== 'undefined') {
  const nkzSdk = (window as any).__NKZ_SDK__;
  if (nkzSdk?.i18n) {
    nkzSdk.i18n.addResources('en', 'weather-map', en);
    nkzSdk.i18n.addResources('es', 'weather-map', es);
    nkzSdk.i18n.addResources('eu', 'weather-map', eu);
    nkzSdk.i18n.addResources('fr', 'weather-map', fr);
    nkzSdk.i18n.addResources('pt', 'weather-map', pt);
    nkzSdk.i18n.addResources('ca', 'weather-map', ca);
  }
}
